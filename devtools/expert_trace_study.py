"""CPU-only, held-out cache replay; no model loading or throughput prediction."""
import argparse
import csv
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

GIB = 1024**3
TRAIN = {"warmup_decode", "prompt0_decode"}


def layer_of(key):
    return int(key.split(".layers.")[1].split(".")[0])


def replay(events, hot, capacity, partitioned=False, details=None):
    caches = defaultdict(OrderedDict)
    pinned = set()
    seen = set()
    phases = defaultdict(Counter)
    for phase, kind, key, _ in events:
        if kind == 1:
            pinned.clear()
            continue
        if kind != 0:
            raise ValueError("study expects access/boundary only")
        stats = phases[phase]
        stats["requests"] += 1
        if details is not None:
            details[key]["requests"] += 1
        if key in hot:
            stats["hot_hits"] += 1
            continue
        cache = caches[layer_of(key) if partitioned else 0]
        if key in cache:
            cache.move_to_end(key)
            stats["dynamic_hits"] += 1
        else:
            stats["misses"] += 1
            stats["reloads"] += int(key in seen)
            if details is not None:
                details[key]["misses"] += 1
                details[key]["reloads"] += int(key in seen)
            if len(cache) == capacity:
                victim = next((k for k in cache if k not in pinned), None)
                if victim is None:
                    raise ValueError("replay capacity cannot hold active GPU leases")
                del cache[victim]
                stats["evictions"] += 1
            cache[key] = None
            seen.add(key)
        pinned.add(key)
    return dict(phases)


def analyze(events):
    sizes = {size for _, kind, _, size in events if kind == 0}
    if len(sizes) != 1 or next(iter(sizes)) <= 0:
        raise ValueError("replay requires equal nonzero expert byte geometry")
    size = sizes.pop()
    training = Counter(key for phase, kind, key, _ in events if kind == 0 and phase in TRAIN)
    heldout = Counter(key for phase, kind, key, _ in events
                      if kind == 0 and phase in {"prompt1_decode", "prompt2_decode"})
    keys = [f"language_model.model.layers.{layer}.mlp/{expert}"
            for layer in range(48) for expert in range(288)]
    known = set(keys)
    if any(kind == 0 and key not in known for _, kind, key, _ in events):
        raise ValueError("unknown expert key")
    ranked = sorted(keys, key=lambda key: (-training[key], key))
    per_layer = []
    ranked256 = set()
    numeric256 = {key for key in keys if int(key.rsplit("/", 1)[1]) < 256}
    for layer in range(48):
        ordered = sorted(keys[layer*288:(layer+1)*288], key=lambda k: (-training[k], k))
        ranked256.update(ordered[:256])
        cold = set(ordered[256:])
        per_layer.append({"layer": layer, "training_decode_requests": sum(training[k] for k in ordered),
                          "heldout_decode_requests": sum(heldout[k] for k in ordered),
                          "ranked_cold32_heldout_requests": sum(heldout[k] for k in cold),
                          "cold32_ids": [int(k.rsplit("/", 1)[1]) for k in ordered[256:]],
                          "training_counts": [training[k] for k in keys[layer*288:(layer+1)*288]],
                          "heldout_counts": [heldout[k] for k in keys[layer*288:(layer+1)*288]]})
    configs = [
        ("capture224_layer8", {key for key in keys if int(key.rsplit("/", 1)[1]) < 224}, 8, True),
        ("numeric256_layer8", numeric256, 8, True),
        ("numeric256_global384", numeric256, 384, False),
        ("ranked256_layer8", ranked256, 8, True),
        ("ranked256_global384", ranked256, 384, False),
        ("ranked16g_dynamic8g", set(ranked[:16*GIB//size]), 8*GIB//size, False),
        ("ranked16g_dynamic12g", set(ranked[:16*GIB//size]), 12*GIB//size, False),
        ("dynamic24g", set(), 24*GIB//size, False),
        ("dynamic28g", set(), 28*GIB//size, False),
    ]
    results = []
    details = defaultdict(Counter)
    test_events = [event for event in events if event[0].startswith(("prompt1_", "prompt2_"))]
    for name, hot, capacity, partitioned in configs:
        phases = replay(events, hot, capacity, partitioned,
                        details if name == "capture224_layer8" else None)
        total = sum((Counter(v) for v in phases.values()), Counter())
        test = sum((Counter(v) for p, v in phases.items() if p.startswith(("prompt1_", "prompt2_"))), Counter())
        cold_test = sum((Counter(v) for v in replay(test_events, hot, capacity, partitioned).values()), Counter())
        results.append({"name": name, "hot_gib": len(hot)*size/GIB,
                        "dynamic_gib": capacity*(48 if partitioned else 1)*size/GIB,
                        "total": dict(total), "heldout": dict(test),
                        "heldout_empty_dynamic": dict(cold_test), "phases": phases})
    for layer in per_layer:
        group = keys[layer["layer"]*288:(layer["layer"]+1)*288]
        layer["capture_misses"] = [details[k]["misses"] for k in group]
        layer["capture_reloads"] = [details[k]["reloads"] for k in group]
    return {"expert_bytes": size, "training_phases": sorted(TRAIN),
            "note": "PP counts grouped accesses, decode counts actual routed selections; replay is not SSD I/O or tok/s",
            "ranked_by_training": ranked, "layers": per_layer, "configs": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-misses", type=int, required=True)
    parser.add_argument("--expected-evictions", type=int, required=True)
    args = parser.parse_args()
    if args.trace.stat().st_size > 64*1024*1024:
        raise ValueError("trace exceeds file-size safety bound")
    with args.trace.open() as stream:
        events = [(r["phase"], int(r["kind"]), r["key"], int(r["bytes"]))
                  for r in csv.DictReader(stream, delimiter="\t")]
    if len(events) > 200000:
        raise ValueError("trace exceeds capture bound")
    required = {"warmup_decode", *(f"prompt{p}_{kind}" for p in range(3) for kind in ("pp", "decode"))}
    if {phase for phase, _, _, _ in events} != required:
        raise ValueError("incomplete or unexpected study phases")
    counts = Counter(phase for phase, kind, _, _ in events if kind == 0)
    if any(counts[p] != 64*48*10 for p in required if p.endswith("decode")):
        raise ValueError("incomplete decode capture")
    result = analyze(events)
    live = result["configs"][0]["total"]
    if live.get("misses", 0) != args.expected_misses or live.get("evictions", 0) != args.expected_evictions:
        raise ValueError("replay does not match live fixed cache")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for cfg in result["configs"]:
        print(cfg["name"], "expert_gib", round(cfg["hot_gib"]+cfg["dynamic_gib"], 3),
              "heldout", cfg["heldout"])


if __name__ == "__main__":
    main()

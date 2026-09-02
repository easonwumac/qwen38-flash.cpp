#!/usr/bin/env python3
"""Small deterministic mixed-domain MTP policy probe."""

import argparse
import hashlib
import json
import urllib.request


PROMPTS = {
    "code": "Write a Python function that merges two sorted integer lists without using sort().",
    "json": "Return JSON with keys name, version, features, and enabled for a local inference server.",
    "explain": "Explain why memory bandwidth limits token generation on a local language model.",
    "creative": "Write a short scene about an astronomer discovering a silent blue comet.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11439")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    results = {}
    for name, prompt in PROMPTS.items():
        body = json.dumps({
            "model": "qwen38-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens,
            "temperature": 0,
            "thinking": False,
            "stream": False,
        }).encode()
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        text = (payload["choices"][0]["message"].get("content") or "") + (
            payload["choices"][0]["message"].get("reasoning_content") or ""
        )
        mtp = payload["performance"]["mtp"]
        results[name] = {
            "completion_tokens": payload["usage"]["completion_tokens"],
            "finish_reason": payload["choices"][0]["finish_reason"],
            "tps": payload["performance"]["generation_tps"],
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rounds": mtp["rounds"],
            "accepted": mtp["accepted"],
            "proposed": mtp["proposed"],
            "final_depth": mtp["depth"],
            "demotions": mtp["demotions"],
            "accepted_by_position": mtp["accepted_by_position"],
            "proposed_by_position": mtp["proposed_by_position"],
            "top2_rejected_by_position": mtp.get("top2_rejected_by_position"),
            "top2_recovered_by_position": mtp.get("top2_recovered_by_position"),
            "draft_ms": mtp.get("draft_ms", 0.0),
            "verify_ms": mtp.get("verify_ms", 0.0),
            "commit_ms": mtp.get("commit_ms", 0.0),
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MTPLX reference trace used to bisect the native 48-layer decode graph."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


def host_sum(value: mx.array) -> float:
    return sum(value.astype(mx.float32).reshape(-1).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    from mtplx.models.qwen4_exp import create_ssm_mask
    from mtplx.runtime import load

    runtime = load(args.model, mtp=False)
    wrapper = runtime.model
    text = wrapper.language_model
    body = text.model
    cache = wrapper.make_cache()
    ids = mx.array([[9419]], dtype=mx.int32)
    hidden = body.embed_tokens(ids)
    ssm_mask = create_ssm_mask(hidden, cache[body.ssm_idx])
    if body._ple_stage_idx is not None:
        ple = body.layers[body._ple_stage_idx].ple
        ple.ple_embedding.stage(ids, cache[body._ple_stage_idx], ple.NGRAM_IDX)
    hidden = mx.tile(hidden, (1, 1, body.args.hc_count))
    checksums = []
    for layer, layer_cache in zip(body.layers, cache):
        hidden = layer(
            hidden,
            input_ids=ids,
            ssm_mask=ssm_mask,
            cache=layer_cache,
        )
        mx.eval(hidden)
        checksums.append(host_sum(hidden))
    mixed = body.hyper_connection_mixer(hidden)
    logits = text.lm_head(mixed)
    mx.eval(logits)
    print(
        json.dumps(
            {
                "layer_checksums": checksums,
                "token": int(mx.argmax(logits[0, -1]).item()),
                "logit": float(mx.max(logits[0, -1]).item()),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()

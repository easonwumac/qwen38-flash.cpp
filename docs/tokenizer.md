# Native tokenizer

The online tokenizer is C++ and uses `utf8proc` only for Unicode NFC
normalization, category lookup, and UTF-8 validation. It independently implements
the checkpoint's regex-equivalent pre-split, GPT-style byte alphabet, ranked BPE
merges, added/special tokens, and byte-level decode.

Parity fixtures were generated once with the checkpoint's pinned Hugging Face
`tokenizer.json` and are committed as fixed expected token IDs. Runtime serving
does not call Python or Hugging Face tokenizers.

The initial implementation prioritizes correctness. Loading the retained 248K
vocabulary measured about 0.44 seconds and 103 MB peak footprint on the reference
Mac. This is small relative to the checkpoint but too large to ignore under the
30-36 GB resident target; a later packed string arena and flat/sorted lookup will
replace the current unordered-map-heavy representation after parity remains fixed.

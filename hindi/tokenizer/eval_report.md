# Tokenizer Evaluation Report: Hindi (HI)

Evaluated on 4,428 held-out test documents from `hindi/data/test.jsonl`.

## Quantitative Benchmark Summary

| Tokenizer | Vocab Size | Fertility (↓) | Bytes/Token (↑) | Chars/Token | Continuation Rate | UNK Rate | Speed (tok/s) | Throughput |
|---|---|---|---|---|---|---|---|---|
| **BPE-5K** | 6832 | **1.55** w | 8.2 B | 3.27 C | 31.03% | 0.0% | 1,193,593 | 9.33 MB/s |
| **BPE-10K** | 11263 | **1.397** w | 9.09 B | 3.63 C | 23.76% | 0.0% | 1,060,313 | 9.2 MB/s |
| **BPE-20K** | 20195 | **1.29** w | 9.85 B | 3.93 C | 18.14% | 0.0% | 979,965 | 9.21 MB/s |

Tokenized sample sentences are saved in `tokenized_sample_sentences.json`.
## Analysis & Observations

1. **Fertility & Subword Compression**: As vocabulary scales from 5K to 20K merges, fertility should generally drop as common roots, inflections, and postpositions are fused into fewer tokens.
2. **Out-of-Vocabulary (OOV) Handling**: The custom BPE tokenizer achieves **0.00% unknown token rate** by maintaining single-character fallback units for the complete Devanagari Unicode range.
3. **Throughput**: Subword tokenization throughput exceeds tens of thousands of tokens per second with rank-based heap optimization.

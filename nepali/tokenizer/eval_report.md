# Tokenizer Evaluation Report: Nepali (NE)

Evaluated on 660,655 held-out test documents from `nepali/data/test.jsonl`.

## Quantitative Benchmark Summary

| Tokenizer | Vocab Size | Fertility (↓) | Bytes/Token (↑) | Chars/Token | Continuation Rate | UNK Rate | Speed (tok/s) | Throughput |
|---|---|---|---|---|---|---|---|---|
| **BPE-5K** | 4777 | **1.731** w | 9.98 B | 3.73 C | 41.81% | 0.0% | 1,019,097 | 9.7 MB/s |
| **BPE-10K** | 9137 | **1.507** w | 11.46 B | 4.28 C | 31.52% | 0.0% | 892,769 | 9.76 MB/s |
| **BPE-20K** | 17973 | **1.353** w | 12.77 B | 4.77 C | 23.27% | 0.0% | 791,549 | 9.64 MB/s |

Tokenized sample sentences are saved in `tokenized_sample_sentences.json`.
## Analysis & Observations

1. **Fertility & Subword Compression**: As vocabulary scales from 5K to 20K merges, fertility should generally drop as common roots, inflections, and postpositions are fused into fewer tokens.
2. **Out-of-Vocabulary (OOV) Handling**: The custom BPE tokenizer achieves **0.00% unknown token rate** by maintaining single-character fallback units for the complete Devanagari Unicode range.
3. **Throughput**: Subword tokenization throughput exceeds tens of thousands of tokens per second with rank-based heap optimization.

# Per-tensor quant mix vs Unsloth IQ4\_XS (Qwen3.5-9B)

25 October 2026 · findings from testing `ggml_quant_inspect.py`

## Summary

A mixed GGUF produced by `ggml_quant_inspect.py --recommend` (imatrix-weighted SQNR, size-budget allocation) has about half the KL divergence of Unsloth's IQ4\_XS on Qwen3.5-9B and runs about 50% faster on an Intel Arc A750. It is also about 40% larger, so the quality gap is not a size-matched result. The mix is being kept as is.

## Setup

- **Model:** Qwen3.5-9B, BF16 GGUF as the reference.
- **Mix:** script with the Unsloth imatrix, candidates q2\_k to q8\_0, `--budget-gib 5.5` for the matmul weights. `token_embd` was kept at BF16 and excluded from the budget, so the file is about 5.5 + 1.89 ≈ 7.4 GiB (about 7.9 GB).
- **Baseline:** Unsloth IQ4\_XS, 5.5 GB including embeddings.
- **Evaluation:** llama.cpp KL-divergence run against BF16 base logits, n\_ctx 2048, same text and settings for both.

## Results

| Metric | Script mix (84 chunks) | Unsloth IQ4\_XS (42 chunks) |
| --- | --- | --- |
| ln(PPL(Q)/PPL(base)) | 0.0075 ± 0.0009 | 0.031 ± 0.002 |
| KL divergence | 0.019 ± 0.001 | 0.0356 ± 0.0022 |
| Δp RMS | 3.7% | 4.8% |
| Same top token | 95.2% | 93.6% |
| File size | about 7.9 GB | 5.5 GB |
| Generation speed | about 30 tok/s | about 20 tok/s |

Both runs had stopped moving when they were cut off, so the averages are stable. The mix's KLD is about 0.53× Unsloth's, and it flips the top token on about 4.8% of tokens against 6.4%.

## Caveats

- **Size is not matched.** The mix is about 2.4 GB larger, and the extra bits (BF16 embedding, q6\_k/q8\_0 on sensitive tensors) account for part of the quality gap. A matched test would pin `token_embd` to q5\_k and set the budget to about 5.1 GiB total. It was not run.
- **Speed is not explained by size.** The embedding is a lookup, not a matmul, so it adds little to generation cost. The likely cause is that the Unsloth file contains tensor types that are slow on the Arc backend (for example BF16 or F16 SSM tensors). This is unverified; counting bytes per tensor type in each file would confirm it.
- **Different chunk counts.** The Unsloth run was stopped at 42 chunks because it is slow on this machine.

## Script changes

- **Done:** imatrix fallback. Tensors with no usable imatrix data (as in the llama-quantize message `did not find weights for ...`) are scored with the plain metric, flagged in the table, and listed in the summary and the emitted command.
- **To do:** do not quantize `token_embd` by default when it is not tied to the output tensor. It is a lookup table, costs the most bytes for no speed benefit, and it was the largest single contributor to file size here.
- **Minor:** the message for the 24 `ssm_conv1d` tensors says llama-quantize picks its own fallback. They stay F32, so the message should say that instead.

## Conclusion

The allocation produced a clearly more accurate and faster model than Unsloth's IQ4\_XS on this machine. How much of the accuracy comes from the allocation itself, and how much from the larger file, remains open until the size-matched run is done.

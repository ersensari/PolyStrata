# Ornith: sharing CPU IQ weight decoding across tokens

Measured 2026-10-21 on one RTX 5060 Ti (host GPU 1 only, 16311 MiB), Ryzen 9 9900X
(12 physical cores), Linux, 91 GiB system RAM, CUDA 13.0 / sm_120. The other GPU was not used.
This branch builds on GQA-16 commit `7fb9d79d`; it does not include the pinned-host KV experiment.

## Change

`STRATA_POLY_IQ_MT=1` opts `strata-poly` into the first engine's existing AVX-512 IQ row kernels for
CPU experts shared by several tokens. Gate and up are addressed separately, including across GGUF shards;
their rows are processed in bounded stack scratch. Down results are kept per expert use and added in the
original router order. Weight formats, activations' quantization, chosen experts and routing weights are unchanged.
Single-token gate/up groups, unsupported formats and CPUs without the required ISA retain ggml's dots.
No GPU kernel, speculation policy, cache allocation, expert skipping or CPU-KV allocation was changed.

The IQ kernels accumulate floats differently from ggml's dots. This is opt-in, and its generated tokens and
MTP acceptance can change. It is not a bit-identical optimization when enabled. Unset, the old arithmetic stays.
Unlike upstream PR 36, this change shares IQ block decoding across tokens rather than adding CPU prefetch.

## Setup

AtomicChat Ornith AD-IQ3_XXS, 31 shards, Q8 GPU KV, configured maximum context 262144, 12 CPU threads,
1024 MiB VRAM reserve, greedy sampling, window 8 / lookup 3, MTP enabled, tail skip 0, expert swaps 0.
Both fast attention opt-ins were on. Every process restored the same operator profile; its SHA256 is
`94b96773251b6b05e303da84b36fa6f78692fed7c7e9079b75bf2f43efc7bb3b`. The profile is not published.
Both arms loaded 1826 / 12544 experts into 6.4 GiB VRAM. Production inference was stopped for the measurements
and restored afterwards. Other host services remained running. Model load is excluded from token throughput.

Baseline executable SHA256: `bcba061330f6ba51718a9c1adc5333d2ec4e4c008b4194a77b176247f78b5bbd`.
Patched executable SHA256: `14a8ef3fb95bad281380c436894687260b7d22b4b373ec13db0a63f113d73521`.

## Results

Three prompts, two repetitions each, alternating on/off order; each generated 256 tokens.
The exact prompts, IDs, engine logs, telemetry and counters are in `compare.py` and `raw/`.
The table shows medians of the two runs per prompt; the aggregate is the median of all six per arm.

| Prompt | Unset tok/s | Enabled tok/s |
| --- | ---: | ---: |
| Python LRU cache | 34.15 | 35.73 |
| Asyncio in Turkish | 34.82 | 36.70 |
| Arithmetic parser | 31.50 | 33.21 |
| All six short runs | 33.97 | 35.57 |
| 7246-token repeated-word context, one pair | 34.22 | 36.62 |

All six short pairs favored the opt-in; aggregate medians differ by 4.7%. The one long-context pair differs
by 7.0%; prompt speed there was 894.8 / 906.8 tok/s. This is a small sample, not a confidence interval or a
prediction for the operator's conversations. Outputs and accepted drafts differ, so this is equal-prompt,
equal-output-budget measurement, not replay of identical model activations. No claim of equal answer quality.

Median short-run CPU-expert counters fell from 22.6 to 21.3 ms/token. These include overlapped work and the
engine's own token accounting; they are not an independent decode-time breakdown. Long-run counters are divided
by prompt tokens too, so should not be compared directly with short-run counters.
GPU telemetry did not show a utilization or power gain: medians of each short run's final 15 samples
(about three seconds at 200 ms intervals) were 56.2% / 52.6% utilization and 85.16 / 85.29 W power. Sampling
is coarse and token streams differ. The result is a modest throughput gain, not evidence of greater GPU occupancy.

## Checks and limits

- Built CUDA `strata-poly`, `poly_workers_test` and `poly_cpu_iq_test`; workers passed 4000 checks.
- No-draft, frozen placement: original and patched/unset generated identical 256 token IDs, SHA256
  `4dd0bb12315d63f1f257fe80cfb0d57aaa4dbb8a35f101e97761f43617052974`.
- `poly_cpu_iq_test FIRST_SHARD`: 64 real-weight cases per run, rows 1..8, shared and distinct experts,
  SwiGLU/GELU, infinite and finite clamps. Enabled IQ3's maximum relative L2 error against separate-token
  ggml calls was 0.00030556808 (0.0306%), below the 0.5% guard. Unset IQ3 and enabled with
  `STRATA_FORCE_ISA=avx2` had zero error. Enabled IQ4 (unsupported type pairs) also had zero error, 64 cases.
- No full 256K run, full HIP/SYCL build, other CPU or GPU architecture, vision workload or quality benchmark.
  IQ2 formats are dispatched through existing kernels but were not exercised with model weights here.
- The comparison engine's CPU IQ changes do not make Qwen and Ornith equivalent workloads. Local metadata:
  Qwen uses 10 experts of width 2560 x 640 per routed layer, Ornith 8 of 3072 x 1024. For the three routed
  projections this is about 54% more active matrix elements per Ornith layer, before other architecture costs.

## Reproduce

Build with the same CUDA/ggml configuration as the GQA-16 report, then:

```sh
cmake --build build --target strata-poly poly_cpu_iq_test poly_workers_test -j8
build/poly_workers_test
STRATA_POLY_IQ_MT=1 build/poly_cpu_iq_test FIRST_SHARD
python bench/results/2026-10-21-ornith-iq-mt/compare.py \
  --model FIRST_SHARD --baseline ORIGINAL_GQA16_ENGINE --engine build/strata-poly \
  --tokenizer TOKENIZER_DIR --profile SEED_PROFILE --out RESULTS
```

Run with only the test GPU visible and production inference stopped. The Python script restores the profile,
warms the OS file cache, checks the unset path, and measures fresh processes with both flag states.

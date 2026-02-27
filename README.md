## Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive)  Decoding?

<div align="center">
Pengxiang Li*<sup>1</sup> &nbsp;&nbsp; Dilxat Muhtar*<sup>2,3,4</sup> &nbsp;&nbsp; Tianlong Chen<sup>6</sup>&nbsp;&nbsp; Lu Yin*<sup>5</sup>&nbsp;&nbsp; Shiwei Liu*<sup>2,3,4</sup>


<sup>1</sup> The Hong Kong Polytechnic University, <sup>2</sup> ELLIS Institute Tübingen, <sup>3</sup> Max Planck Institute for Intelligent Systems, <sup>4</sup> Tübingen AI Center, <br>
<sup>5</sup> University of Surrey, <sup>6</sup> The University of North Carolina at Chapel Hill

</div>

<div align="center">
[<a href="https://arxiv.org/pdf/2602.23225">Paper</a>] | [<a href="">Blog</a>]
</div>
<br>

## Overview

Diffusion Language Models (DLMs) are often described as naturally parallel decoders. In practice, many decoding trajectories still collapse into autoregressive (AR-like) left-to-right behavior.

This project presents **NAP** (**N**on-**A**utoregressive **P**arallel DLMs), a data-decoding co-design framework:

1. **Parallel data curation**: each training sample contains multiple independent reasoning trajectories, not a single privileged chain.
2. **Parallel-forced decoding**: generation updates are enforced across multiple reasoning streams at every decoding step.

The goal is to reduce AR-like decoding bias while preserving (or improving) reasoning accuracy.

## Key Diagnosis

- Common corpora (FineWeb, OpenR1-Math) are strongly sequential.
- AO (confidence-based arbitrary-order) decoding in standard DLMs still shows high ARness.
- Long-CoT SFT further increases ARness.
- Existing fast DLM decoding methods often improve speed by following an AR-like critical path.

## TODOs
We will try our best to achieve
- \[✅\] Training code of NAP
- \[🚀\] Datasets and Model weights
- \[🚀\] Inference and evaluation code

## Method

NAP uses a structured output canvas:

```text
[<think #1>, R^(1), <think #2>, R^(2), ..., <think #m>, R^(m), <summary>, S]
```

- `R^(j)` are independent reasoning paths.
- `<summary>` aggregates evidence from all paths and outputs the final answer.

Decoding policy:
<img width="1312" height="452" alt="Image" src="https://github.com/user-attachments/assets/451c2b13-e8ca-439e-ab95-087bfd7e8ffc" />
- **Macro-parallel**: distribute unmasking budget across all reasoning blocks each step.
- **Micro-confidence**: within each block, commit tokens by confidence (not strict left-to-right order).

## Main Quantitative Results

### GSM8K (Accuracy, %)

| Steps | Tok/Step | LLaDA Long-CoT | NAP-LLaDA | Gain | Dream Long-CoT | NAP-Dream | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4 | 54.1 | **56.1** | +2.0 | 46.5 | **60.9** | +14.4 |
| 336 | 3 | 60.9 | **63.3** | +2.4 | 56.9 | **70.9** | +14.0 |
| 512 | 2 | 82.0 | **82.6** | +0.6 | 66.8 | **79.2** | +12.4 |
| 1024 | 1 | 83.5 | **84.1** | +0.6 | 78.0 | **83.6** | +5.6 |

### MATH-500 (Accuracy, %)

| Steps | Tok/Step | LLaDA Long-CoT | NAP-LLaDA | Gain | Dream Long-CoT | NAP-Dream | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4 | 21.4 | **26.6** | +5.2 | 16.2 | **23.8** | +7.6 |
| 336 | 3 | 26.6 | **35.4** | +8.8 | 25.6 | **31.4** | +5.8 |
| 512 | 2 | 41.2 | **43.0** | +1.8 | 40.0 | **43.0** | +3.0 |
| 1024 | 1 | 45.0 | **47.0** | +2.0 | 47.4 | **49.6** | +2.2 |

### GPQA (Accuracy, %)

| Steps | Tok/Step | LLaDA Long-CoT | NAP-LLaDA | Gain | Dream Long-CoT | NAP-Dream | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 336 | 3 | 15.4 | **19.0** | +3.6 | 7.3 | **10.5** | +3.2 |
| 512 | 2 | 21.2 | **25.9** | +4.7 | 19.4 | **22.5** | +3.1 |
| 1024 | 1 | 23.0 | **28.6** | +5.6 | 28.6 | **29.5** | +0.9 |

### ARness Findings

- **AO decoding ARness (Global-ARness@1):**
  - LLaDA-8B: `0.73`
  - Dream-7B: `0.92`
- **After Long-CoT SFT:**
  - LLaDA-8B: `0.73 -> 0.81` (`+0.08`)
  - Dream-7B: `0.92 -> 0.93` (`+0.01`)

These numbers indicate that standard supervision tends to increase autoregressive decoding behavior.


## Citation

```bibtex
```

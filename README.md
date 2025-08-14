# iSafetyBench: A Video-Language Benchmark for Safety in Industrial Environments [VISION'25 Workshop - ICCVW'25]

[![Website](https://img.shields.io/badge/Project-Website-blue)](https://isafetybench.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2508.00399)

🎉 **(July 12, 2025)** Paper accepted at the VISION'25 Workshop - ICCVW'25

## Dataset
Dataset videos are available on [Hugging Face](https://huggingface.co/datasets/raiyaanabdullah/isafety-bench/tree/main).  
The list of actions and video annotations are provided in this repository.

## Running Evaluation
The following instructions are for evaluating Ovis2-8B, the top-performing model on our dataset. Other models can be evaluated using a similar script.

1. Set up the environment by following the [official Ovis2 instructions](https://github.com/AIDC-AI/Ovis).
2. Download the `mcq` questions from [this repository](https://github.com/iSafetyBench/data/tree/main/mcq).
3. Update the file paths in the `evaluate_ovis.py` script, available [here](https://github.com/iSafetyBench/data/blob/main/evaluate_ovis.py).
4. Run the script. Our setup used an NVIDIA RTX A6000 (48 GB) GPU.

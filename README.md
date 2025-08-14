# iSafetyBench: A Video-Language Benchmark for Safety in Industrial Environments [VISION'25 Workshop - ICCVW'25]

[![Website](https://img.shields.io/badge/Project-Website-blue)](https://isafetybench.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2508.00399)

🎉 **(July 12, 2025)** Paper accepted at the VISION'25 Workshop - ICCVW'25

## Dataset
Dataset videos are available on [Hugging Face](https://huggingface.co/datasets/raiyaanabdullah/isafety-bench/tree/main).  
The list of actions and video annotations are provided in this repository.

## Performance of VLMs on normal and danger/hazard scenarios
|                     | **Normal** |                  |                   |                   | **Danger/Hazard** |                  |                   |                   | **Average of Both** |                   |
|---------------------|:----------:|:----------------:|:-----------------:|:-----------------:|:-----------------:|:----------------:|:-----------------:|:-----------------:|:-------------------:|:-----------------:|
| **Model**           | **Single** | **Multi**        | **Multi**         | **Multi**         | **Single**        | **Multi**        | **Multi**         | **Multi**         | **Single**          | **Multi**         |
|                     | **Acc (%)**| **Precision**    | **Recall**        | **F1 Score**      | **Acc (%)**       | **Precision**    | **Recall**        | **F1 Score**      | **Acc (%)**         | **F1 Score**      |
|---------------------|------------|------------------|-------------------|-------------------|-------------------|------------------|-------------------|-------------------|---------------------|-------------------|
| **Ovis2-8B**       | 47.3       | 47.6             | **71.3**          | **53.4**          | **40.3**          | 45.0             | 54.1              | 46.2              | **43.8**            | 49.8              |
| **InternVL2.5-8B-MPO** | 42.2   | 47.2             | 64.1              | 50.8              | 38.3              | 47.0             | **57.7**          | **49.0**          | 40.25               | **49.9**          |
| **Qwen2.5-VL-7B-Instruct** | 46.9 | 44.5           | 62.7              | 49.2              | 33.6              | 40.5             | 47.9              | 41.7              | 40.25               | 45.45             |
| **VideoLLaMA3-7B** | 38.9       | 49.6             | 37.4              | 39.7              | 32.7              | 46.2             | 34.1              | 36.5              | 35.8                | 38.1              |
| **VideoChat-Flash-7B** | 31.0   | 36.3             | 44.2              | 33.6              | 26.8              | 36.0             | 39.1              | 31.0              | 28.9                | 32.3              |
| **Oryx-7B**        | 25.0       | 37.8             | 41.7              | 30.3              | 21.5              | 34.7             | 37.2              | 26.6              | 23.25               | 28.45             |
| **Valley-Eagle-7B**| **48.8**   | **59.1**         | 47.7              | 48.7              | 35.9              | **54.5**         | 36.8              | 40.9              | 42.35               | 44.8              |
| **GPT-4o**         | 40.3       | 50.0             | 59.1              | 51.6              | 37.3              | 49.3             | 45.4              | 45.1              | 38.8                | 48.35             |

## Running Evaluation
The following instructions are for evaluating Ovis2-8B, the top-performing model on our dataset. Other models can be evaluated using a similar script.

1. Set up the environment by following the [official Ovis2 instructions](https://github.com/AIDC-AI/Ovis).
2. Download the `mcq` questions from [this repository](https://github.com/iSafetyBench/data/tree/main/mcq).
3. Update the file paths in the `evaluate_ovis.py` script, available [here](https://github.com/iSafetyBench/data/blob/main/evaluate_ovis.py).
4. Run the script. Our setup used an NVIDIA RTX A6000 (48 GB) GPU.

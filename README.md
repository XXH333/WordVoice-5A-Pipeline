# WordVoice Data Pipeline 🚀

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arxiv_2026-blue.svg)](#)
[![DemoPage](https://img.shields.io/badge/DemoPage-WordVoice-yellow.svg)](https://xxh333.github.io/wordvoice-demo/)
[![Dataset](https://img.shields.io/badge/Dataset-WordVoice--5A-green.svg)](https://huggingface.co/datasets/XXH333/WordVoice-5A)
[![Model](https://img.shields.io/badge/Model-WordVoice-red.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official linguistically-guided annotation pipeline for the WordVoice dataset**

</div>

---

## Overview

**WordVoice Data Pipeline** is the official annotation toolkit for constructing **large-scale, high-quality, word-level acoustic datasets** for controllable LLM-based Text-to-Speech (TTS).

Unlike conventional forced-alignment pipelines that only provide timestamps, our framework integrates **dual-model alignment**, **boundary refinement**, and **multi-dimensional prosodic annotation**, enabling the automatic construction of datasets such as **WordVoice-5A**.

The pipeline is language-aware and currently supports both **Mandarin Chinese** (`zh`) and **English** (`en`).

---

## ✨ Features

### 🎯 Accurate Word Alignment

- Dual-model alignment using:
  - **Montreal Forced Aligner (MFA)**
  - **Qwen3FA**
- Automatic consistency checking between two aligners
- Confidence filtering for unreliable alignments

### 🔍 Boundary Refinement

- Loudness-based word boundary optimization
- Removes excessive silence
- Mitigates coarticulation bleeding
- Produces more natural acoustic segments

### 📊 Five-Dimensional Acoustic Annotation

Each word is annotated with five complementary acoustic attributes:

| Feature | Description |
|----------|-------------|
| ⏱️ Duration | Word-level duration |
| ⏸️ Boundary | Five-level pause category (`b0`–`b4`) |
| 🔊 Energy | Truncated & normalized syllable nucleus energy |
| 🎵 Pitch | Core F0 extraction with bilateral truncation |
| 📈 Tone | Seven-category prosodic morphology via 16-point polynomial regression |

### 🌍 Bilingual Support

- ✅ Mandarin Chinese (`zh`)
- ✅ English (`en`)

---

# Installation

Because **Montreal Forced Aligner (MFA)** depends on Kaldi and several C++ libraries, we strongly recommend installing it with **Conda**.

## Step 1. Create Environment

```bash
conda create -n wordvoice-5a python=3.10 -y
conda activate wordvoice-5a
```

## Step 2. Install MFA

```bash
conda install -c conda-forge montreal-forced-aligner=3.3.8 -y
```

## Step 3. Clone Repository

```bash
git clone https://github.com/yourusername/wordvoice-data-pipeline.git

cd wordvoice-data-pipeline
```

## Step 4. Install Dependencies

```bash
pip install qwen-asr

pip install -e .
```

---

# Download Models

Download all required models by running:

```bash
bash download_models.sh
```

This script automatically downloads all dependencies required by the pipeline, including the alignment models and related resources.

---

# Run the Pipeline

A complete example is provided in:

```text
test_demo/
```

The directory contains sample audio and transcription files that can be used to quickly verify the installation and reproduce the pipeline outputs.

Example:

```text
test_demo/
├── audio_files/
└── json_files/
```

To test the pipeline directly:

```bash
bash data.sh
```

---


# Supported Languages

| Language | Status |
|-----------|--------|
| Mandarin Chinese | ✅ |
| English | ✅ |

Additional languages will be supported in future releases.

---

# Citation

If you use this project or the **WordVoice** dataset, please cite our paper.

```bibtex
@article{wordvoice2026,
  title={WordVoice: Linguistically-Guided Word-Level Acoustic Dataset for Controllable LLM-based Text-to-Speech},
  author={Anonymous},
  journal={arXiv},
  year={2027}
}
```

---

# License

This project is released under the MIT License.

See [LICENSE](LICENSE) for details.

---

# Acknowledgements

This project builds upon several outstanding open-source projects, including:

- Montreal Forced Aligner (MFA)
- Qwen3FA

We sincerely thank the authors and contributors of these projects.

---

# Contact

For questions, bug reports, or collaboration opportunities, please open an Issue or submit a Pull Request.

---

<div align="center">

**WordVoice Data Pipeline**

Building high-quality word-level acoustic annotations for controllable speech generation.

</div>

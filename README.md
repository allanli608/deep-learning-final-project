# Lost in Translation: Why Zero-Shot Transfer Fails at Detecting Subjectivity in Chinese

**Authors:** 李盼嘉乐 (Allan Li) et al.
**Institution:** Tsinghua University

**Final Report:** [Lost in Translation: Why Zero-Shot Transfer Fails at Detecting Subjectivity in Chinese](report.pdf)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) [![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## 📖 Abstract
Neutral Style Transfer—the automated removal of subjective bias (e.g., changing *"The radical regime"* to *"The government"*)—is a well-solved task in English but remains difficult for low-resource languages like Chinese due to a lack of parallel corpora.

This repository contains the code and datasets for our report. We investigate whether large multilingual models (mBART-50) can "transfer" the semantic concept of subjectivity from English to Chinese without target-language supervision.

**Key Findings:**
1.  **Zero-Shot Transfer Fails:** Without target-language supervision, models default to "copying" input, failing to detect bias (53% Accuracy).
2.  **Pivot Translation Destroys Meaning:** Translating `Zh -> En -> De-bias -> Zh` removes bias but corrupts semantic details (0.835 Similarity).
3.  **Synthetic Data Wins:** Training on noisy, machine-translated data (our "Silver Standard") achieves the best balance of Neutrality and Semantic Preservation.

---

## 🚀 Installation

```bash
# Clone the repository
git clone [https://github.com/allanli608/chinese-neutral-style-transfer.git](https://github.com/allanli608/chinese-neutral-style-transfer.git)

# Install dependencies
conda env create --file environment_gpu.yml # substitute this for cpu to avoid downloading large libraries

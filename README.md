<h1 align="center">MISP-QEKS Baseline</h1>

<p align="center">
Official Baseline Implementation for<br>
<b>MISP-QEKS: A Large-Scale Tri-Modal Benchmark for Query-by-Example Keyword Spotting</b><br>
ACM Multimedia 2025
</p>

<p align="center">
<img src="https://img.shields.io/badge/Task-Tri--modal%20QEKS-blue" />
<img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
<img src="https://img.shields.io/badge/License-Apache%202.0-green" />
</p>

---

## 🔥 Overview

This repository provides the official baseline implementation for the MISP-QEKS benchmark.

MISP-QEKS introduces the first **Text–Audio–Visual** open-vocabulary Query-by-Example Keyword Spotting (QEKS) benchmark with:

- 610,000 enrollment–query pairs  
- 9,830+ distinct keywords  
- In-Vocabulary (IV) & Out-of-Vocabulary (OOV) splits  
- Real-world noise simulation  

📦 Dataset available at:  
👉 [https://huggingface.co/Igor97/MISP-QEKS](https://huggingface.co/datasets/Igor97/MISP-QEKS)

The baseline implements:

- **XEQ-Matcher** – Cross-modal enrollment–query matcher  
- **VGM** – Visual Gating Module (noise suppression)  
- **MAM** – Multimodal Alignment Module (representation alignment)  

Supports:

- Text–Audio–Visual enrollment  
- Audio–Visual query  
- Robust matching under noisy conditions  

---

## 📂 Repository Structure


library/ # Shared utilities
loader/ # Data loading pipeline
model/ # Pretrained feature extractors
res/ # Experimental logs

train.py # Training entry
test.py # Evaluation entry

run_train.sh # Training script
run_test.sh # Testing script


---

## ⚙️ Environment Setup

### Requirements

- Python >= 3.8  
- PyTorch >= 1.10  
- CUDA >= 11.3  

Install dependencies:

```bash
pip install -r requirements.txt

📦 Dataset Preparation

Download MISP-QEKS from HuggingFace and organize as:

data/
  train/
  dev/
  eval_seen/
  eval_unseen/

Ensure paths in run_train.sh and run_test.sh are correctly configured.

🚀 Quick Start
🔹 Train from Scratch
bash run_train.sh
🔹 Evaluate
bash run_test.sh
🔹 Evaluate with Pretrained Checkpoint

The official 10-epoch checkpoint is available in the dataset repository:

HuggingFace → train/model/

python test.py --ckpt path/to/model_epoch10.pth
📊 Baseline Performance
XEQ-Matcher (Full Tri-modal)
Split	AUC (%)	EER (%)
Eval-seen	82.82	24.23
Eval-unseen	79.79	26.20
+ VGM + MAM
Split	AUC (%)	EER (%)
Eval-seen	85.94	21.60
Eval-unseen	85.44	21.49
🎯 Training Details

Batch size: 64

Optimizer: SGD

Learning rate: 0.01

α = 0.01 (VGM loss weight)

β = 0.5 (MAM loss weight)

Binary Cross-Entropy objective

4 × Tesla V100 (32GB)

🔁 Reproducibility Notes

To reproduce reported results:

Use provided pretrained encoders in model/

Maintain 1:4 positive–negative ratio

Respect speaker-independent split

Use SNR levels {+5, 0, −5, −10} dB

No external data allowed

🧩 Evaluation Protocol

Separate evaluation on:

Eval-seen (IV)

Eval-unseen (OOV)

Metrics:

AUC

EER

The evaluation strictly follows the protocol described in the ACM MM 2025 paper.

📜 Citation

If you use this repository, please cite:

@inproceedings{xiong2025mispqeks,
  title={MISP-QEKS: A Large-Scale Dataset with Multimodal Cues for Query-by-Example Keyword Spotting},
  author={Xiong, Shifu and Chen, Hang and others},
  booktitle={ACM Multimedia},
  year={2025}
}

🛡 License

This project is released under the Apache 2.0 License.

<p align="center"> Built for Robust Multimodal Keyword Spotting Research </p> ```


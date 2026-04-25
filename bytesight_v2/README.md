# ByteSight v2: Hierarchical Malware Analysis

This folder contains the professional refactor of ByteSight, implementing a high-performance, 2-stage hierarchical classification pipeline.

## Key Features

### 1. Two-Stage Hierarchical Pipeline

ByteSight v2 implements a standard security industry workflow:

- **Stage 1 (Binary Detector):** A specialized "Gatekeeper" trained to distinguish between **Safe (Windows System Files)** and **Unsafe (Malicious)** binaries.
- **Stage 2 (Family Classifier):** A "Profiler" that identifies the specific malware family (e.g., *Ramnit*, *Gatak*, *Kelihos*) only if the sample is flagged as malicious in Stage 1.

### 2. Real-World Benign Integration

Uses actual Windows binaries (`C:\Windows\System32` and `SysWOW64`) as the benign class. This ensures the model learns the texture of production operating system code rather than simple Linux binaries.

### 3. Professional Evaluation Suite

Training automatically generates metrics saved to the `checkpoints/` folder:

- **Confusion Matrix Heatmaps:** Visualize model confusion across all families.
- **Classification Reports:** Industry-standard Precision, Recall, and F1-Scores.
- **Lossless Padding:** 100% data integrity preservation before ResNet input.

---

## 🛠 Usage Guide

### 1. Data Preparation

Run these from the **root** project directory to generate the image datasets:

```bash
python bytesight_v2/process_microsoft_challenge.py --cores 12
python bytesight_v2/prepare_benign_dataset_v2.py --limit 10868 --cores 12
```

### 2. Training the Pipeline

From the `bytesight_v2` folder, train both stages:

```bash
python v2_main.py train --mode binary --epochs 15
python v2_main.py train --mode malware_only --epochs 15
```

### 3. Inference & Demo

Run a single binary / png through the full hierarchical pipeline:

```bash
python v2_main.py predict --input path/to/sample.exe --gradcam
```

Or run a random demo on the test set:

```bash
python v2_main.py demo --num_samples 20
```

---

## 📂 Architecture

- **`models/resnet_model.py`**: Modular ResNet wrapper for 1-channel binary textures.
- **`dataset.py`**: Advanced data loader supporting Binary and Family-Only mapping.
- **`v2_main.py`**: Unified Command Line Interface for ByteSight.

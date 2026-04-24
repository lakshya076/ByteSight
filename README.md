# ByteSight

ByteSight is an advanced **Binary-to-Image Malware Classifier**. It visualizes executable binaries as grayscale images and uses a deep ResNet-18 Convolutional Neural Network (CNN) to identify malicious patterns and family signatures.

## 🚀 ByteSight v2 (Current Focus)
The project has transitioned to a professional, high-performance architecture located in the `bytesight_v2/` directory.

### Key Improvements in v2:
*   **Hierarchical Pipeline:** A 2-stage system featuring a **Binary Detector** (Safe vs. Unsafe) and a **Malware Family Classifier**.
*   **Zero-Loss Integrity:** Uses lossless padding to preserve 100% of binary data during image conversion.
*   **Windows Native Benign Set:** Eliminates classification bias by using actual Windows system binaries as the benign class.
*   **Professional Reporting:** Automated generation of Confusion Matrices and Classification Reports.

## 📂 Project Structure
*   **`bytesight_v2/`**: The core application, including training, modular models, and inference CLI.
*   **`microsoft_dataset/`**: (Generated) Standardized image dataset organized for training.
*   **`checkpoints/`**: (Generated) Model weights and evaluation reports.

## 📥 Model Weights
Model weights (`.pth` files) are excluded from the repository to keep it lightweight. Please refer to the **GitHub Releases** section to download pre-trained weights for the hierarchical pipeline.

---
For detailed setup and training instructions, please see the [ByteSight v2 README](bytesight_v2/README.md).

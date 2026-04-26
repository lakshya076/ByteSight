# ByteSight

Binary-to-Image Malware Classifier. It visualizes executable binaries as grayscale images and uses a deep ResNet-18 Convolutional Neural Network (CNN) to identify malicious patterns and family signatures.

Built using Pytorch.

Currently uses two datasets

1. root directory is trained and built upon [malimg](https://www.kaggle.com/datasets/manmandes/malimg)
2. `bytesight_v2/` directory is built upon [Microsoft Malware Challenge](https://www.kaggle.com/c/malware-classification)

## ByteSight v1

This version contains only the image-to-malware classifier since the dataset used ([malimg](https://www.kaggle.com/datasets/manmandes/malimg)) is a pre-built kaggle dataset of images of malware executables.

Download the dataset into your working directory using

```bash
curl -L -o ~/ByteSight/dataset https://www.kaggle.com/api/v1/datasets/download/manmandes/malimg
```

After this run

```bash
unzip dataset
```

The dataset doesn't contain benign images so it can't classify between benign and malware. It only classifies the classes of malware. To inject benign images run the command

```bash
python3 prepare_benign_dataset.py --input "/mnt/c/Windows/System32" --limit 600
```

This command will take 600 binary/executable files from the mentioned directory (System32 for me) and convert them to images and inject in the dataset folder as benign images for training the network.

### How to run

If you wish to run the pre-trained model before starting your own training run the command

```python3
python3 demo.py --temperature 2.0
```

Adjust the temperature flag to get softer confidence scores instead of a hard 100% which appears usually when the classification is correct.

## ByteSight v2

For detailed setup and training instructions using Microsoft Dataset, please see the [ByteSight v2 README](bytesight_v2/README.md).

### Key Improvements in v2

* **Hierarchical Pipeline:** A 2-stage system featuring a **Binary Detector** (Safe vs. Unsafe) and a **Malware Family Classifier**.
* **Zero-Loss Integrity:** Uses lossless padding to preserve 100% of binary data during image conversion.
* **Windows Native Benign Set:** Eliminates classification bias by using actual Windows system binaries as the benign class.
* **Professional Reporting:** Automated generation of Confusion Matrices and Classification Reports.

## 📂 Project Structure

* **`./`**: Root directory. Contains the initial version trained on [malimg](https://www.kaggle.com/datasets/manmandes/malimg) dataset.
* **`./bytesight_v2/`**: The core application, including training, modular models, and inference CLI. Uses [Microsoft Malware Challenge](https://www.kaggle.com/c/malware-classification) dataset.
* **`./malimg_dataset/`**: (Unarchived) The resultant dataset when malimg dataset is downloaded and uncompressed.
* **`./microsoft_dataset/`**: (Generated) Standardized image dataset organized for training. It contains the images (converted from binary files from the Microsoft Dataset download) and the Benign files from the host computer.
* **`./checkpoints/`**: (Generated) Model weights and evaluation reports for v2.

## 📥 Model Weights

Model weights (`.pth` files) are excluded from the repository to keep it lightweight. Please refer to the **GitHub Releases** section to download pre-trained weights for the hierarchical pipeline.

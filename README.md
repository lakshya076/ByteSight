# ByteSight

Binary-to-Image Classifier. ByteSight aims to read executable binary files and then classify them as recognised malware files by converting to grayscale images and feeding it to a Resnet18 CNN.

Built using Pytorch.

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

## How to run

If you wish to run the pre-trained model before starting your own training run the command

```python3
python3 demo.py --temperature 2.0
```

Adjust the temperature flag to get softer confidence scores instead of a hard 100% which appears usually when the classification is correct.

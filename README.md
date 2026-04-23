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

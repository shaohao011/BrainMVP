# BrainMVP - Official PyTorch Implementation

## Dependencies
+ Python 3.9.19
+ PyTorch 2.0.1
## Usage of the pre-trained BrainMVP

### 1. Clone the repository
```bash
$ cd BrainMVP/
$ pip install -r requirements.txt
```
### 2. Download the pre-trained BrainMVP

| Weight | Download | Description |
|  ----  | ----  |  ----  |
| BrainMVP_unet.pt  | [link]() | pre-trained U-Net weights |
| BrainMVP_uniformer.pt | [link]() | pre-trained Uniformer weights |
### 3. Model structure
We provide detailed model structure of U-Net and Uniformer used in our paper at ./Fine_tune, and you can finetune pre-trained model provided above on your own tasks.
# Pre-training
Code will soon be available.
# Finetune
Code will soon be available.
# Acknowledgement
We thank [MONAI](https://monai.io/) for their excellent framework for medical image analysis.

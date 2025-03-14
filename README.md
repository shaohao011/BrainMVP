# [BrainMVP: Multi-modal Vision Pre-training for Medical Image Analysis (CVPR 2025)](https://arxiv.org/abs/2410.10604)

- stay tuned, we are updating recently.
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
| BrainMVP_unet.pt  | [link](https://drive.google.com/drive/folders/1ZNcIznsPhQHIVM_vx3QqiEybfSh9ZJE6?usp=sharing) | pre-trained U-Net weights |
| BrainMVP_uniformer.pt | [link](https://drive.google.com/drive/folders/1ZNcIznsPhQHIVM_vx3QqiEybfSh9ZJE6?usp=sharing) | pre-trained Uniformer weights |
Note the link is empty now, we'll release the pre-trained weights upon request for information leakage concern.
### 3. Model structure
We provide detailed model structure of U-Net and Uniformer used in our paper at ./Fine_tune, and you can finetune pre-trained model provided above on your own tasks.
# Pre-training
We'll release the codes upon request.
# Finetune
We'll release the codes upon request.
# Acknowledgement
We thank [MONAI](https://monai.io/) for their excellent framework for medical image analysis.

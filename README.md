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
| BrainMVP_unet.pt  | [link](https://drive.google.com/file/d/1-54c_VChYVa2bFB_6VJh1bRItckqOF-o/view?usp=drive_link) | pre-trained U-Net weights |
| BrainMVP_uniformer.pt | [link](https://drive.google.com/file/d/1o3pPEHeCEhRaDjqtufJMq6W6F40_sdxY/view?usp=drive_link) | pre-trained Uniformer weights |

Download the pre-trained BrainMVP and save into `./pretrained_weights` directory.
### 3. Load Pre-trained model
We have provided code examples for loading our BrainMVP pre-trained weights in classification tasks in [UNET3D.py](Fine_tune\classification\models\UNET3D.py) and [Uniformer.py](Fine_tune\classification\models\Uniformer.py).
# Finetune
our finetune codes will soon be available
# Acknowledgement
We thank [MONAI](https://monai.io/) for their excellent framework for medical image analysis.

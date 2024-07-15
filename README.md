# BrainMVP - Official PyTorch Implementation


## Dependencies

+ Linux
+ Python 2.7+
+ PyTorch 1.3.1
## Usage of the pre-trained BrainMVP

### 1. Clone the repository
```bash
$ git clone https://github.com/shaohao011/BrainMVP.git
$ cd BrainMVP/
$ pip install -r requirements.txt
```
### 2. Download the pre-trained Models Genesis

| Weight | Download | Description |
|  ----  | ----  |  ----  |
| BrainMVP_unet.pt.h5  | [link](https://huggingface.co/MrGiovanni/ModelsGenesis/resolve/main/Genesis_Chest_CT.h5?download=true) | pre-trained U-Net weights |
| BrainMVP_uniformer.pt | [link](https://huggingface.co/MrGiovanni/ModelsGenesis/resolve/main/Genesis_Chest_CT.pt?download=true) | pre-trained Uniformer weights |

Download the pre-trained BrainMVP and save into `./pretrained_weights` directory.

### 3. Fine-tune BrainMVP on your own target task
BrainMVP learn a general-purpose image representation that can be leveraged for a wide range of target tasks. Specifically, BrainMVP can be utilized to initialize the encoder for the target <i>classification</i> tasks and to initialize the encoder-decoder for the target <i>segmentation</i> tasks.

As for the target classification tasks, the 3D deep model can be initialized with the pre-trained encoder using the following example:
```bash
$ cd classification 
$ bash do_train_cls.sh
```

As for the target segmentation tasks, the 3D deep model can be initialized with the pre-trained encoder-decoder using the following example:
```bash
$ cd segmentation 
$ bash do_train_seg.sh
```

**Prepare your own data:** If the image modality in your target task is CT, we suggest that all the intensity values be clipped on the min (-1000) and max (+1000) interesting Hounsfield Unit range and then scale between 0 and 1. If the image modality is MRI, we suggest that all the intensity values be clipped on min (0) and max (+4000) interesting range and then scale between 0 and 1. For any other modalities, you may want to first clip on the meaningful intensity range and then scale between 0 and 1. 

We adopt input cubes shaped in (N, 1, 64, 64, 32) during model pre-training, where N denotes the number of training samples。 When fine-tuning the pre-trained Models Genesis, **any arbitrary input size** is acceptable as long as it is divisible by 16 (=2^4) due to four down-sampling layers in V-Net. That said, to segment larger objects, such as liver, kidney, or big nodule :-( you may want to try
```python
input_channels, input_rows, input_cols, input_deps = 1, 128, 128, 64
input_channels, input_rows, input_cols, input_deps = 1, 160, 160, 96
```
or even larger input size as you wish.


## Learn Models Genesis from your own unlabeled data

### 1. Clone the repository
```bash
$ git clone https://github.com/MrGiovanni/ModelsGenesis.git
$ cd ModelsGenesis/
$ pip install -r requirements.txt
```
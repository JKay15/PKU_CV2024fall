# Nueral Style Transfer and Anto-Mask Pyramid STROTSS Method

This readme file gives the user basic guidance to our project. All the programs are running in Jupter Notebook.

## 1.Basic Installation

### 1.1 Environments

All the packages used in program are included in **requirements.txt**.

Under the master directory, open your command prompt and start your own virtual environment, run command **`pip install -r requirements.txt`** , then you can have all the packages installed.

### 1.2 Models

The Gatys and LapStyle program use the VGG19 model used by Gatys in 2014, which you can install from our link.

The improvements use the VGG19 model which Pytorch has provided, and will be installed automatically when run the file.

## 2. Usage

### 2.1 Gatys

The code is in **lap_style_transfer.ipynb**, to run Gatys's method, you should check the weight of loss_lap in function **closure** and set it to zero. Then you can run the whole file and see the results in it. Also, various hyperparameters can be changed as you wanted, like weights and iterations.

### 2.2 LapStyle

The code is in **lap_style_transfer.ipynb**, to run LapStyle method, you should check the weight of loss_lap in function **closure** and set it to 100 (recommended) . Then you can run the whole file and see the results in it. Also, various hyperparameters can be changed as you wanted, like weights and iterations.

### 2.3 AMP STROTSS

You can run our prior method of Patch Match in **Patch_Match_Example.ipynb**.

To run our improvements, you should put **Patch_Match.py** and **strotss_auto_mask_pyramid.ipynb** in the same directory, then set the args (hyperparameter) in **stross_auto_mask_pyramid.ipynb** to below:

```python
args={"content": "content.jpg", "style": "style.jpg", "content_mask": None, "style_mask": None, "weight": 1.0, "output": "strotss.png", "device": "cuda:0", "ospace": "uniform", "resize_to": 512,"AUTO_MASK_PYRAMID":False}
```

Here, "content" and "style" are the paths of the content image and style image respectively; "content_mask" and "style_mask" are the paths of the content and style masks respectively; "weight" is the proportion of content in the loss function, and it is recommended to be 1.0; "output" is the name of the output image; "device" indicates on which device PyTorch runs; "ospace" can be set to "uniform" by default and does not need to be managed; "resize_to" is the length to which the longer side of the image is resized, and it is recommended to be 512 for convenient use of Pyramid; "AUTO_MASK_PYRAMID" is a variable that sets whether to use auto_mask_pyramid. If it is True, then it is auto_mask_pyramid; otherwise, it is the general STROTSS. You can change them as you wanted.

Then you can run the file and see results in it.

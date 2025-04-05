# Convolutional Neural Networks (CNNs) for Image Classification on CINIC10 dataset

## Overview
This repository contains implementations of various Convolutional Neural Network (CNN) architectures for image classification tasks, specifically designed to work with the CINIC10 dataset. The models are implemented using PyTorch and are trained on the CINIC10 dataset, which is a large-scale image classification dataset containing 270,000 images across 10 classes.

## How to train the model?

1. Clone the repository:
```bash
git clone https://github.com/AKapich/Convolutional_Neural_Network.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the CINIC10 dataset:
(for example, [here](https://www.kaggle.com/datasets/mengcius/cinic10)) and place it in the `data` directory.

4. Run the training script:
```bash
python train.py --model <model_name> --n_epochs <epochs> --model_config <model_config> 
```

The training process may be adjusted with flags to set. 
Below there's an overview:
- `--model`: The type of the model to train (`resnet`, `vgg16` or `custom` model based on GoogLeNet). 
- `--n_epochs`: The number of epochs to train the model. Default is 10.
- `--model_config`: The configuration JSON file for the model. It's responsible for setting simple training and regularization parameters (check out `example_config.json` to gain better understanding)
- `--data_augmentation`: Data augmentation techniques to apply during training. Options include 'None`, 'rotation', 'flip', `color_jitter`, `cutout` and `combined` which uses all of them except for `cutout`.
- `--hyperparameter_optimization`: Whether to perform hyperparameter optimization (it's a boolean flag). If set, grid search will be performed.
- `resize`: Whether images resized from 32x32 to 224x224 ought to be used.

5. Gather the results 
The results will be saved in the `trained_models` directory. The results contain model checkpoints along with the log files for monitoring training process. 


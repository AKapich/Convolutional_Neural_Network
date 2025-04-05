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


### Few Shot Learning

For few shot learning, the `train_few_shot.py` script can be used. It allows training a model with few shot learning techniques. The script accepts different parameters than the aforementioned training script.

The parameters are as follows:
- `--model`: The type of the model to train (`resnet`, `vgg16` or `custom` model based on GoogLeNet).
- `--n_epochs`: The number of epochs to train the model. Default is 10. Each epoch consists of `N` training episodes, whereas `N` is set in the `--n_training_episodes` parameter.
- `--n_way`: The number of classes in a task. Default is 5.
- `--n_shot`: Number of images per class in the support set. Default is 4.
- `--n_query`: Number of images per class in the query set. Default is 5.
- `--n_evaluation_tasks': Number of tasks to evaluate the model. Default is 50.
- `--n_train_tasks`: Number of tasks to train the model. Default is 100.
- `--n_training_episodes`: Number of training episodes. Default is 100.
- `--checkpoint_path`: Path to the checkpoint file.

The numbers were set low since originally the script was  run on CPU.

The command to run:

```bash
python train_few_shot.py --model <model_name> --n_epochs <epochs> --n_way <n_way> --n_shot <n_shot> --n_query <n_query> --n_evaluation_tasks <n_evaluation_tasks> --n_train_tasks <n_train_tasks> --n_training_episodes <n_training_episodes> --checkpoint_path <checkpoint_path>
```


### Model evaluation

To evaluate the model, you can use the `evaluate.py` script. This script allows you to test the trained model on the CINIC10 dataset and obtain accuracy metrics.

Parameters to be passed are 
- `--model`: The type of the model to evaluate (`resnet`, `vgg16` or `custom` model based on GoogLeNet).
- `--model_path`: Path to the trained model checkpoint.
- `--output_path`: Path to save the evaluation results.

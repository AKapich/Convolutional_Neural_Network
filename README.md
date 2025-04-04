# Convolutional Neural Networks (CNNs) for Image Classification on CINIC10 dataset

How to train the model?

1. Clone the repository:
```bash
git clone https://github.com/AKapich/Convolutional_Neural_Network.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the CINIC10 dataset:
Download it [here](https://www.kaggle.com/datasets/mengcius/cinic10) and place it in the `data` directory.

4. Run the training script:
```bash
python train.py --model <model_name> --n_epochs <epochs> --model_config <model_config> 
```


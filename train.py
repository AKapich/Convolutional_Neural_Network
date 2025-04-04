from argparse import ArgumentParser
import torch
from src.models.our_model import OurModel
from src.models.pretrained_models import VGG16Pretrained, ResNetPretrained
from src.model_trainer import ModelTrainer
from src.utils import load_data
from src.transformations import normalized_simple_transform
import os
import itertools as it
import logging
import json

torch.manual_seed(123)


training_process_grid = {
    "optimizer_type": ["adam", "sgd"],
    "learning_rate": [0.01, 0.001, 0.0001],
}

regularization_grid = {
    "weight_decay": [0.0001, 0.001, 0.01],
    "dropout": [0.2, 0.5, 0.7],
}

model_mapping = {
    "vgg16": VGG16Pretrained,
    "resnet": ResNetPretrained,
    "custom": OurModel,
}


def main(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")
    torch.set_num_threads(2)

    def initialize_model(model_name, model_config, device):
        if model_name == "custom":
            return model_mapping[model_name](dropout=model_config["dropout"])
        return model_mapping[model_name](
            device=device, num_classes=model_config["num_classes"]
        )

    train_loader = load_data(
        os.path.join(os.path.dirname(__file__), "data", "train"),
        batch_size=128,
        shuffle=True,
        transform=normalized_simple_transform(),
        num_workers=1,
    )
    valid_loader = load_data(
        os.path.join(os.path.dirname(__file__), "data", "valid"),
        batch_size=512,
        shuffle=True,
        transform=normalized_simple_transform(),
        num_workers=1,
    )

    logging.info(f"Training model: {args.model}")
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    if args.hyperparameter_optimization:
        training_param_combinations = list(it.product(*training_process_grid.values()))
        regularization_param_combinations = (
            list(it.product(*regularization_grid.values()))
            if args.model == "custom"
            else regularization_grid["weight_decay"]
        )

        logging.info("Searching for optimal training hyperparameters")
        for optimizer_type, learning_rate in training_param_combinations:
            model = initialize_model(args.model, model_config, device)
            logging.info(
                f"\n\nTraining with optimizer: {optimizer_type}, learning rate: {learning_rate}"
            )
            learning_rate_str = f"{learning_rate}".replace("0.", "")
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                device=device,
                weight_decay=0,
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                log_file=f"{args.model}_{optimizer_type}_{learning_rate}.json",
                save_dir=f"./trained_models/{args.model}/optimization/training_params_{optimizer_type}_{learning_rate_str}",
                max_batches=None,
                valid_loader=valid_loader,
            )
            trainer.train(args.n_epochs)

        logging.info("Searching for optimal regularization hyperparameters")
        for params in regularization_param_combinations:
            if args.model == "custom":
                weight_decay, dropout = params
                save_dir = f"./trained_models/{args.model}/optimization/regularization_params_{weight_decay}_{dropout}"
                log_file = f"{args.model}_{weight_decay}_{dropout}.json"
            else:
                weight_decay, dropout = params, ""
                save_dir = f"./trained_models/{args.model}/optimization/regularization_params_{weight_decay}"
                log_file = f"{args.model}_{weight_decay}.json"

            logging.info(
                f"\n\nTraining with weight decay: {weight_decay}, dropout: {dropout}"
            )

            model = initialize_model(args.model, model_config, device)

            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                device=device,
                weight_decay=weight_decay,
                optimizer_type="sgd",
                learning_rate=0.001,
                log_file=log_file,
                save_dir=save_dir,
                max_batches=None,
                valid_loader=valid_loader,
            )
            trainer.train(args.n_epochs)

    elif not args.hyperparameter_optimization:
        logging.info(
            f"\n\nTraining model {args.model} with default parmeters derived from {args.model_config}"
        )
        model = initialize_model(args.model, model_config, device)

        weight_decay = model_config["weight_decay"]
        optimizer_type = model_config["optimizer_type"]
        learning_rate = model_config["learning_rate"]

        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            device=device,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            log_file=f"{args.model}_{optimizer_type}_{learning_rate}.json",
            save_dir=f"./trained_models/{args.model}",
            max_batches=None,
            valid_loader=valid_loader,
        )
        trainer.train(args.n_epochs)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--model",
        choices=["vgg16", "resnet", "custom"],
        required=True,
        help="Architecture used in the training process",
        default="resnet",
    )
    argparser.add_argument(
        "--hyperparameter_optimization",
        action="store_true",
        help="Boolean flag to enable hyperparameter optimization instead of casual training",
    )
    argparser.add_argument(
        "--model_config",
        type=str,
        default="example_config.json",
        help="Path to the configuration file for the model",
    )
    argparser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        required=True,
        help="Number of epochs for training",
    )

    main(args=argparser.parse_args())

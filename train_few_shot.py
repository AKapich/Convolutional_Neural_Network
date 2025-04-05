from argparse import ArgumentParser
from src.few_shot import *
from torchvision import datasets
from src.transformations import *
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from src.models.our_model import *
import logging
import os

torch.manual_seed(123)

model_mapping = {
    "vgg16": vgg16(weights=VGG16_Weights.DEFAULT),
    "resnet": resnet18(weights=ResNet18_Weights.DEFAULT),
    "custom": OurModel,
}


def main(args):
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query
    n_training_episodes = args.n_training_episodes
    n_evaluation_tasks = args.n_evaluation_tasks
    n_validation_tasks = args.n_validation_tasks

    train_set = datasets.ImageFolder(root="./data/train", transform=resize_transform())
    test_set = datasets.ImageFolder(root="./data/test", transform=resize_transform())
    validation_set = datasets.ImageFolder(
        root="./data/valid", transform=resize_transform()
    )

    train_loader = get_few_shot_dataloader(
        train_set, n_way, n_shot, n_query, n_training_episodes
    )
    test_loader = get_few_shot_dataloader(
        test_set, n_way, n_shot, n_query, n_evaluation_tasks
    )

    validation_loader = get_few_shot_dataloader(
        validation_set, n_way, n_shot, n_query, n_validation_tasks
    )

    logging.info(f"Chosen model: {args.model}")
    logging.info(f"Training model with {n_way} way, {n_shot} shot, {n_query} query")
    model = model_mapping[args.model]
    model.classifier = nn.Flatten()

    fewshot_trainer = FewShotTrainer(
        model=PrototypicalNetwork(model),
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=validation_loader,
        checkpoint_path=f"{args.checkpoint_path}/{args.model}/best_{args.model}.pt",
    )

    checkpoint_dir = f"{args.checkpoint_path}/{args.model}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fewshot_trainer.train(
        num_epochs=args.n_epochs,
        log_save_file=f"{args.checkpoint_path}/{args.model}/{args.model}_log",
    )

    fewshot_trainer.evaluate()


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
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs for training. One epoch consists of N training episodes (set as another parameter)",
    )
    argparser.add_argument(
        "--n_way",
        type=int,
        default=5,
        help="Number of classes in a task (N-way classification)",
    )
    argparser.add_argument(
        "--n_shot",
        type=int,
        default=4,
        help="Number of images per class in the support set (K-shot learning)",
    )
    argparser.add_argument(
        "--n_query",
        type=int,
        default=5,
        help="Number of images per class in the query set",
    )
    argparser.add_argument(
        "--n_evaluation_tasks",
        type=int,
        default=50,
        help="Number of tasks for evaluation",
    )
    argparser.add_argument(
        "--n_training_episodes",
        type=int,
        default=100,
        help="Number of episodes/iterations for training",
    )
    argparser.add_argument(
        "--n_validation_tasks",
        type=int,
        default=50,
        help="Number of tasks for validation",
    )
    argparser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoint file for the model",
        default="trained_models/few_shot",
    )
    args = argparser.parse_args()
    main(args=args)

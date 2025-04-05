from argparse import ArgumentParser
from src.models.our_model import OurModel
from src.models.pretrained_models import VGG16Pretrained, ResNetPretrained
import torch
from src.utils import load_data, evaluate_model
from src.transformations import normalized_simple_transform
import json
import logging

torch.manual_seed(123)
torch.set_num_threads(14)


model_mapping = {
    "vgg16": VGG16Pretrained,
    "resnet": ResNetPretrained,
    "custom": OurModel,
}


def main(args):

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    test_loader = load_data(
        "./data/test",
        batch_size=128,
        shuffle=True,
        transform=normalized_simple_transform(),
        num_workers=1,
    )

    model = model_mapping[args.model](device=device)

    model.load_state_dict(
        torch.load(
            f"./{args.model_path}",
            map_location=device,
        )
    )
    results = evaluate_model(model=model, dataloader=test_loader, device=device)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(
        f"{results['accuracy']:.2f}%, {results['f1_score']:.2f}, {results['roc_auc']:.2f} "
    )
    logging.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--model",
        choices=["vgg16", "resnet", "custom"],
        required=True,
        help="Architecture type to be tested",
        default="resnet",
    )
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./output")

    main(args=argparser.parse_args())

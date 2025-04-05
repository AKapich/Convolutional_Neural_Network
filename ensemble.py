from argparse import ArgumentParser
import torch
from src.models.our_model import OurModel
from src.models.pretrained_models import VGG16Pretrained, ResNetPretrained
from src.models.ensemble import HardVotingEnsemble, SoftVotingEnsemble
from src.utils import load_data, evaluate_model
from src.transformations import normalized_simple_transform
import json
import logging

torch.manual_seed(123)
torch.set_num_threads(2)


def main(args):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    test_loader = load_data(
        "./data/test",
        batch_size=256,
        shuffle=True,
        transform=normalized_simple_transform(),
        num_workers=1,
    )

    vgg16 = VGG16Pretrained(device="cpu")
    vgg16.load_state_dict(
        torch.load(
            args.vgg_model_path,
            map_location=device,
        )
    )
    vgg16.to(device)

    custom = OurModel(aux_enabled=False, se_squeeze=8)
    custom.load_state_dict(torch.load(args.custom_model_path, map_location=device))
    custom.to(device)

    resnet = ResNetPretrained()
    resnet.load_state_dict(torch.load(args.resnet_model_path, map_location=device))
    resnet.to(device)

    base_models = [custom, vgg16, resnet]
    ensemble = (
        HardVotingEnsemble(base_models)
        if args.ensemble_type == "hard"
        else SoftVotingEnsemble(base_models)
    )
    ensemble.to(device)
    results = evaluate_model(ensemble, test_loader, device)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--resnet_model_path", type=str, required=True)
    argparser.add_argument("--vgg_model_path", type=str, required=True)
    argparser.add_argument("--custom_model_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./output")
    argparser.add_argument(
        "--ensemble_type", type=str, choices=["hard", "soft"], default="hard"
    )

    main(args=argparser.parse_args())

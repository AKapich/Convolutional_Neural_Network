import torchvision
from torch.utils.data import DataLoader
import torch
import numpy as np

gen = torch.Generator()
gen.manual_seed(123)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def seed(worker_id):
    np.random.seed(42 + worker_id)


def load_data(
    path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    transform=None,
    num_workers: int = 8,
) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=gen,
        worker_init_fn=seed,
    )


class AugmentedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, transform_augment, augment_prob=0.3):
        super().__init__(root, transform=None)
        self.transform = transform
        self.transform_augment = transform_augment
        self.augment_prob = augment_prob

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if np.random.random() < self.augment_prob:
            image = self.transform_augment(image)
        else:
            image = self.transform(image)
        return image, target


class AugmentedImageFolderPretrained(AugmentedImageFolder):
    def __init__(self, root, transform, transform_augment, augment_prob=0.3):
        super().__init__(
            root,
            transform=transform,
            transform_augment=transform_augment,
            augment_prob=augment_prob,
        )

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        image = self.transform(image)
        if np.random.random() < self.augment_prob:
            image = self.transform_augment(image)
        return image, target


def load_data_augmented_pretrained(
    path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    transform=None,
    augmentation_transform=None,
    augment_prob: float = 0.3,
    num_workers: int = 8,
) -> DataLoader:
    dataset = AugmentedImageFolderPretrained(
        root=path,
        transform=transform,
        transform_augment=augmentation_transform,
        augment_prob=augment_prob,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def load_data_augmented(
    path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    transform=None,
    augmentation_transform=None,
    augment_prob: float = 0.3,
    num_workers: int = 8,
) -> DataLoader:
    dataset = AugmentedImageFolder(
        root=path,
        transform=transform,
        transform_augment=augmentation_transform,
        augment_prob=augment_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=gen,
        worker_init_fn=seed,
    )


def evaluate_model(
    model,
    dataloader,
    device,
    max_batches: int = None,
    metrics_to_compute=["accuracy", "f1_score", "roc_auc"],
) -> dict:
    model.eval()
    all_labels = []
    all_predict = []
    all_prob = []
    batch = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if max_batches is not None and batch == max_batches:
                break
            if batch % 10 == 0:
                print(batch, end=" ")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if outputs.ndim == 1:
                predictions = outputs
                probabilities = None
            else:
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predict.extend(predictions.cpu().numpy())
            if probabilities is not None:
                all_prob.extend(probabilities.cpu().numpy())
            batch += 1

    all_labels = np.array(all_labels)
    all_predict = np.array(all_predict)
    all_prob = np.array(all_prob)

    acc = accuracy_score(all_labels, all_predict)
    f1 = f1_score(all_labels, all_predict, average="macro")

    try:
        roc_auc = roc_auc_score(
            all_labels, all_prob, multi_class="ovr", average="macro"
        )
    except Exception as e:
        roc_auc = None

    return {"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model was saved to {file_path}")


def load_model(model, file_path, device):
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model was read {file_path}")
    return model

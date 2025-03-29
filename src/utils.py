import torchvision
from torch.utils.data import DataLoader
import torch
import numpy as np
gen = torch.Generator()
gen.manual_seed(123)

def seed(worker_id):
    np.random.seed(42 + worker_id)
    
def load_data(path: str, batch_size: int = 32, shuffle: bool = False, transform=None, num_workers: int = 8) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, generator=gen, worker_init_fn=seed)


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
        super().__init__(root, transform=transform, transform_augment=transform_augment, augment_prob=augment_prob)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        image = self.transform(image)
        if np.random.random() < self.augment_prob:
            image = self.transform_augment(image)
        return image, target

def load_data_augmented_pretrained(path: str, batch_size: int = 32, shuffle: bool = False, transform=None, augmentation_transform=None, 
                          augment_prob: float = 0.3, num_workers: int = 8) -> DataLoader:
    dataset = AugmentedImageFolderPretrained(root=path, transform=transform, transform_augment=augmentation_transform, augment_prob=augment_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_data_augmented(path: str,batch_size: int = 32,shuffle: bool = False, transform = None,augmentation_transform = None, augment_prob: float = 0.3, num_workers: int = 8) -> DataLoader:
    dataset = AugmentedImageFolder(root=path, transform=transform, transform_augment=augmentation_transform, augment_prob=augment_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True, generator=gen, worker_init_fn=seed)

def evaluate_model(
    model, dataloader: DataLoader, device: torch.device, max_batches: int = None
) -> float:
    """
    computes accuracy for a model on a test set
    """
    model.eval()
    correct = 0
    total = 0
    batch = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if max_batches:
                if batch == max_batches:
                    break
            if batch % 10 == 0:
                print(batch, end=' ')
            batch += 1 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            max_values, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model was saved to {file_path}")


def load_model(model, file_path, device):
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model was read {file_path}")
    return model

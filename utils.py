
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


def load_data(path: str, batch_size: int = 32, shuffle: bool = False, transform=None, num_workers: int = 8) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def evaluate_model(model, dataloader: DataLoader, device: torch.device, max_batches: int = None) -> float:
    """
    computes accuracy for a model
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
                batch += 1  
            print(batch, end=' ')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
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
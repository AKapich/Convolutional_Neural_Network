"""
Reference: https://github.com/sicara/easy-few-shot-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
import torch.optim as optim
from easyfsl.utils import sliding_average


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
    ):
        super(PrototypicalNetwork, self).__init__()
        self.base_model = base_model

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.base_model.forward(support_images)
        z_query = self.base_model.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        scores = -torch.cdist(z_query, z_proto)
        return scores


def get_few_shot_dataloader(
    data_set: datasets.folder.ImageFolder,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_evaluation_tasks: int,
):

    def get_labels(self):
        return [label for _, label in self.samples]

    data_set.get_labels = get_labels.__get__(data_set)

    test_sampler = TaskSampler(
        data_set,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_evaluation_tasks,
    )

    loader = DataLoader(
        data_set,
        batch_sampler=test_sampler,
        num_workers=0,  # OTHERWISE DOES NOT WORK
        collate_fn=test_sampler.episodic_collate_fn,
    )

    return loader


class FewShotTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion=nn.CrossEntropyLoss(),
        device: torch.device = torch.device("cpu"),
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        patience: int = 5,
    ):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.patience = patience
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0

    def fit(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()
        classification_scores = self.model(support_images, support_labels, query_images)

        loss = self.criterion(classification_scores, query_labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        log_update_frequency = 10
        all_loss = []

        self.model.train()
        with tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                loss_value = self.fit(
                    support_images, support_labels, query_images, query_labels
                )
                all_loss.append(loss_value)

                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(
                        loss=sliding_average(all_loss, log_update_frequency)
                    )
        self.validate()

    def evaluate_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> tuple[int, int]:
        """
        Returns the number of correct predictions of query labels, and the total number of predictions.
        """
        return (
            torch.max(
                self.model(support_images, support_labels, query_images).detach().data,
                1,
            )[1]
            == query_labels
        ).sum().item(), len(query_labels)

    def evaluate(self) -> None:
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        self.model.eval()
        with torch.no_grad():
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
            ) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                support_images = support_images.cpu()
                support_labels = support_labels.cpu()
                query_images = query_images.cpu()
                query_labels = query_labels.cpu()

                correct, total = self.evaluate_on_one_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                )

                total_predictions += total
                correct_predictions += correct

        print(
            f"Model tested on {len(self.test_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
        )

    def validate(self) -> None:
        """
        Evaluate the model on the validation set after each training epoch.
        """
        total_predictions = 0
        correct_predictions = 0

        self.model.eval()
        with torch.no_grad():
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                support_images = support_images.cpu()
                support_labels = support_labels.cpu()
                query_images = query_images.cpu()
                query_labels = query_labels.cpu()

                correct, total = self.evaluate_on_one_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                )

                total_predictions += total
                correct_predictions += correct

        val_accuracy = 100 * correct_predictions / total_predictions
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.patience:
            print("Early stopping triggered. Training stopped.")
            return True

        return False

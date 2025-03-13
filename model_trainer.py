import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import json

class ModelTrainer:
    def __init__(self, model, optimizer, train_loader, device,
                 criterion=None, scaler=None, max_batches=None, log_file=None):
        """
        Używamy GradScaler dla AMP w celu przyspieszenia obliczeń.
        
        model: Model, który będziemy trenować.
        optimizer: Optymalizator używany do trenowania.
        train_loader: Dataloader z danymi treningowymi.
        device: Urządzenie, na którym odbywa się trenowanie.
        criterion: Funkcja straty. Domyślnie nn.CrossEntropyLoss().
        scaler : Obiekt GradScaler dla AMP. Jeśli None, zostanie utworzony.
        max_batches: Maksymalna liczba batchy przetwarzanych w każdej epoce.
                                        Jeśli None, używamy całego zbioru.
        log_file: Ścieżka do pliku, w którym zapiszemy dane z treningu (np. JSON).
                                      Jeśli None, log nie jest zapisywany do pliku.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scaler = scaler if scaler is not None else GradScaler()
        self.max_batches = max_batches
        self.log_file = log_file
        self.training_log = [] 

    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0

            for images, labels in self.train_loader:
                if self.max_batches is not None and batch_count >= self.max_batches:
                    break
                batch_count += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast(device_type=str(self.device), enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0

            epoch_log = {"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc}
            self.training_log.append(epoch_log)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if self.log_file is not None:
                with open(self.log_file, "w") as f:
                    json.dump(self.training_log, f, indent=4)
                print(f"Training log saved to {self.log_file}")
                
    

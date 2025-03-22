import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import re
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self, model, train_loader, device='cpu', optimizer_type='adam', learning_rate=0.001, optimizer_params=None,
                 loss_function=None, max_batches=None, log_file=None, save_dir='.', valid_loader=None):
        """
        model: model to train
        train_loader: dataloader
        device: cpu or cuda
        optimizer_type: adam or sgd
        learning_rate: learning rate
        optimizer_params: custom parameters for optimalizator
        loss_function: loss function
        max_batches: maximum number of batches to train for one epoch (used for code testing only)
        log_file: name of the log file, will be saved in save_dir, if None the log is not saved
        save_dir: directory to save log file and model files
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = loss_function if loss_function is not None else nn.CrossEntropyLoss()
        self.max_batches = max_batches
        self.log_file = log_file
        self.training_log = []
        self.save_dir = save_dir
        self.epoch = 0
        self.valid_loader = valid_loader
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        if optimizer_params is None:
            optimizer_params = {}
            
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=learning_rate, **optimizer_params)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(trainable_params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    def save_log(self):
        if self.log_file is not None:
            path = os.path.join(self.save_dir, self.log_file)
            try:
                with open(path, "w") as file:
                    json.dump(self.training_log, file, indent=4)
                    print(f"Training log saved to {path}")
            except Exception as e:
                print(f"Could not save training log to {path}")
           
            
    def save_model(self, model_file_suffix):
        filename = type(self.model).__name__
        path = os.path.join(self.save_dir, f"{filename}_{model_file_suffix}.pth")
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Could not save model to {path}")
            
    def load_checkpoint(self):
        last_epoch = 0
        if self.log_file:
            log_path = os.path.join(self.save_dir, self.log_file)
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    logs = json.load(f)
                    last_epoch = logs[-1].get("epoch", 0)
                    print(f"Loaded logs from {log_path} with last epoch {last_epoch}")
            else:
                print(f"File {log_path} does not exists")
                
        self.training_log = logs
        self.epoch = last_epoch

        model_class = self.model.__class__.__name__
        checkpoint_file = os.path.join(self.save_dir, f"{model_class}_{last_epoch}.pth")
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model with epoch {self.epoch} from {checkpoint_file}")
        else:
            print(f"Could not find model checkpoint for epoch {last_epoch} in {self.save_dir}")
            
    def compute_loss(self, outputs, labels):
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
            loss_main = self.criterion(main_output, labels)
            loss_aux = self.criterion(aux_output, labels)
            return loss_main + 0.3 * loss_aux
        else:
            return self.criterion(outputs, labels)

    
    def train(self, epochs=1):
        """
        epochs: number of epochs to run
        """
        self.model.train()
        total_epochs = self.epoch + epochs
        for epoch in range(epochs):
            loss_val = 0
            correct = 0
            total = 0
            batch_count = 0
            self.epoch += 1
            
            all_labels = []
            predictions = []


            for images, labels in self.train_loader:
                if batch_count % 10 == 0:
                    print(f"Batch count {batch_count}")
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                loss_val += loss.item() *images.size(0)
                _, predicted = torch.max(main_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
            
                batch_count += 1
                if self.max_batches is not None and batch_count >= self.max_batches:
                    break

            epoch_loss = loss_val / total if total > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            epoch_f1_score = f1_score(all_labels, predictions, average='macro') if total > 0 else 0


            self.training_log.append({"epoch": self.epoch, "loss": epoch_loss, "accuracy": epoch_accuracy, "epoch_f1_score": epoch_f1_score})
            print(f"Epoch {self.epoch}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1-score: {epoch_f1_score:.4f}")
            
            if self.valid_loader:
                loss_val, acc_val, f1score_val = self.compute_valid_loss()
                print(f"Valid Loss: {loss_val:.4f}, Accuracy: {acc_val:.4f}, F1-score: {f1score_val:.4f}")
                self.training_log[-1].update({"loss_val": loss_val,"acc_val": acc_val,"f1score_val": f1score_val})
                
            self.save_log()
            
            model_file_suffix = str(self.epoch)
            self.save_model(model_file_suffix)
            
    def compute_valid_loss(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        predictions = []

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                loss = self.compute_loss(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(main_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())

        avg_loss = val_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        f1_score = f1_score(all_labels, predictions, average='macro') if total > 0 else 0

        return avg_loss, accuracy, f1_score


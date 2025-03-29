import torch
import torch.nn as nn
import torch.nn.functional as F

class HardVotingEnsemble(nn.Module):
    def __init__(self, models):
        """
        models: already trained models
        """
        super(HardVotingEnsemble, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
            
    def forward(self, x):
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                predictions = torch.argmax(output, dim=1)
                all_predictions.append(predictions)
            all_predictions = torch.stack(all_predictions)  
            majority_vote, indx = torch.mode(all_predictions, dim=0)
        return majority_vote

class SoftVotingEnsemble(nn.Module):
    def __init__(self, models):
        """
        models: already trained models
        """
        super(SoftVotingEnsemble, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
            
    def forward(self, x):
        probs_list = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                probs = F.softmax(output, dim=1)
                probs_list.append(probs)
            avg_probs = torch.mean(torch.stack(probs_list), dim=0)
            final_preds = torch.argmax(avg_probs, dim=1)
        return final_preds

class MetaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        :param input_dim: wymiar wejścia (liczba modeli * liczba klas)
        :param num_classes: liczba wyjściowych klas
        """
        super(MetaClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class StackingEnsemble(nn.Module):
    def __init__(self, models, meta_model):
        """
        models: already trained models
        meta_model: meta classifer
        """
        super(StackingEnsemble, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
        self.meta_model = meta_model

    def forward(self, x):
        features = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                feat = F.softmax(output, dim=1)
                features.append(feat)
            all_features = torch.cat(features, dim=1)
        out_meta = self.meta_model(all_features)
        return torch.argmax(out_meta, dim=1)


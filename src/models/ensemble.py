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
                if model.__class__.__name__ in ["VGG16Pretrained", "ResNetPretrained"]:
                    x_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                else:
                    x_input = x
                output = model(x_input)
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
                if model.__class__.__name__ in ["VGG16Pretrained", "ResNetPretrained"]:
                    x_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                else:
                    x_input = x
                output = model(x_input)
                if isinstance(output, tuple):
                    output = output[0]
                probs = F.softmax(output, dim=1)
                probs_list.append(probs)
            avg_probs = torch.mean(torch.stack(probs_list), dim=0)
            final_preds = torch.argmax(avg_probs, dim=1)
        return final_preds



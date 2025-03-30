import os
import json
import matplotlib.pyplot as plt

def plot_metric_from_folders(folders_map, metric='accuracy', metric_type="train"):
    """
        plots logs for any json file in the specified folders
        folders_map: keys are folder paths and values are names shown on the plot
        metric:
        metric_type: train or valid
    """
    plt.figure(figsize=(5, 3))
    
    if metric == 'accuracy':
        if metric_type == "train":
            metric_key = 'accuracy'
        else:
            metric_key = 'acc_val'
    elif metric == 'f1-score':
        if metric_type == "train":
            metric_key = 'epoch_f1_score'
        else:
            metric_key = 'f1score_val'
    elif metric == 'loss':
        if metric_type == "train":
            metric_key = 'loss'
        else:
            metric_key = 'loss_val'
    
    for folder, label in folders_map.items():
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        if not json_files:
            print(f"W folderze {folder} nie znaleziono plików JSON, pomijam.")
            continue
        json_path = os.path.join(folder, json_files[0])
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        epochs = [entry['epoch'] for entry in data]
        values = [entry[metric_key] for entry in data]
        
        plt.plot(epochs, values, marker='o', label=label)
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparizon of {metric_type} {metric} for different parameters')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
    

def plot_metrics(json_path, metric="accuracy"):
    """
    compares metrics on training and validation set 
    metric: accuracy, f1-score, loss
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    epochs = [entry['epoch'] for entry in data]
    
    if metric == "accuracy": 
        train = [entry['accuracy'] for entry in data]
        val = [entry['acc_val'] for entry in data]
    elif metric == "f1-score":
        train = [entry['epoch_f1_score'] for entry in data]
        val = [entry['f1score_val'] for entry in data]
    elif metric == "loss":
        train = [entry['loss'] for entry in data]
        val = [entry['loss_val'] for entry in data]
        
    
    plt.figure(figsize=(5, 3))
    plt.plot(epochs, train, marker='o', label=f'{metric} training')
    plt.plot(epochs, val, marker='o', label=f'{metric} validation')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric}')
    plt.title(f"{metric} changes")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
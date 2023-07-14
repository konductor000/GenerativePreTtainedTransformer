import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


class TestModel:
    def __init__(self):
        self.metrics = {
            'precision': [], 
            'recall': [], 
            'f1': [], 
            'accuracy': [], 
            'loss': []
        }
        self.epochs = []
        self.fig = None
        self.ax = None

    def __call__(self, model, test_dataloader, num_classes):
        return self.test_model(model, test_dataloader, num_classes)
        
    def compute_metrics(self, predicted, true_labels, num_classes):
        # Perform one-hot encoding
        predicted_labels = predicted.argmax(dim=1).numpy()

        # Compute precision, recall, and F1 score
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

        # Compute accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predicted, true_labels)

        return precision, recall, f1, accuracy, loss.item()

    def test_model(self, model, test_dataloader, num_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_predicted = []
        all_true_labels = []

        model.eval()

        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            model_outputs = model(input_ids)

            # Collect predicted and true labels
            all_predicted.append(model_outputs)
            all_true_labels.append(labels)

        # Concatenate and flatten the predicted and true labels
        all_predicted = torch.cat(all_predicted, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)

        # Call compute_metrics once
        precision, recall, f1, accuracy, loss = self.compute_metrics(all_predicted, all_true_labels, num_classes)

        metrics_data = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'loss': loss
        }

        return metrics_data

        
        #self.add_metrics_data(metrics_data)

        #self.visualize_metrics()
    
    def visualize_metrics(self):
        # Get the current axes
        ax = plt.gca()
        
        # Clear the previous plot
        ax.clear()
        
        # Plot each metric
        for metric_name, metric_values in self.metrics.items():
            ax.plot(self.epochs, metric_values, label=metric_name)
        
        # Add labels, title, and legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metrics')
        ax.set_title('Model Evaluation Metrics')
        ax.legend()
        
        # Display the updated plot
        plt.show()
    
    def add_metrics_data(self, metrics_data):
        # Update metrics and epochs data
        for metric_name, metric_value in metrics_data.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(metric_value)
        
        self.epochs.append(len(self.epochs) + 1)

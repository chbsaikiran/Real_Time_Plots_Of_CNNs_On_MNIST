import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import io
import base64
import asyncio

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split training data into train and validation (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Use larger batch size for validation to speed up computation
    val_batch_size = batch_size * 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Use a subset of validation data (20% of validation set)
    val_subset = torch.utils.data.Subset(val_dataset, 
                                       indices=torch.randperm(len(val_dataset))[:len(val_dataset)//5])
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

async def train_batch(model, data, target, optimizer, criterion, device):
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    _, predicted = output.max(1)
    total = target.size(0)
    correct = predicted.eq(target).sum().item()
    
    return loss.item(), (100. * correct / total)

async def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # This ensures no gradients are computed during validation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * target.size(0)  # Weight loss by batch size
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Free memory
            del output, pred
    
    return val_loss / total, 100. * correct / total

def create_plot_data(train_losses, train_accuracies, val_losses, val_accuracies,
                    model_name, progress, epoch, total_epochs, batch, total_batches,
                    current_loss, current_acc, current_val_loss, current_val_acc):
    return {
        "model": model_name,
        f"progress{model_name[-1]}": progress,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "current_loss": current_loss,
        "current_acc": current_acc,
        "current_val_loss": current_val_loss,
        "current_val_acc": current_val_acc
    }

def get_random_test_samples(test_loader, num_samples=10):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    indices = torch.randperm(len(images))[:num_samples]
    return images[indices], labels[indices]

def calculate_confusion_matrix(true_labels, pred_labels):
    """Calculate confusion matrix for MNIST (10 classes)"""
    matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1
    return matrix

def calculate_metrics(confusion_matrix):
    """Calculate accuracy, precision, recall, and F1 score"""
    # Calculate accuracy
    total = confusion_matrix.sum()
    correct = np.diag(confusion_matrix).sum()
    accuracy = correct / total

    # Calculate per-class metrics and average them
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(10):  # For each class
        true_positive = confusion_matrix[i][i]
        false_positive = confusion_matrix[:, i].sum() - true_positive
        false_negative = confusion_matrix[i, :].sum() - true_positive
        
        # Calculate precision
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        precisions.append(precision)
        
        # Calculate recall
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        recalls.append(recall)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores)
    }
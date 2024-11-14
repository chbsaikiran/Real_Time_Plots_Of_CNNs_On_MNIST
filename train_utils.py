import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import io
import base64
import asyncio
import PIL

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Optimize data loading
    kwargs = {
        'num_workers': 2,
        'pin_memory': False,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'drop_last': True
    }
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split training data into train and validation (95-5 split)
    train_size = int(0.95 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Use larger batch sizes for validation
    val_batch_size = min(batch_size * 8, 1024)  # Cap the maximum batch size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    # Ensure validation set has at least one batch
    min_val_samples = max(val_batch_size, len(val_dataset) // 10)  # At least one batch or 10% of validation data
    val_subset = torch.utils.data.Subset(val_dataset, 
                                       indices=torch.randperm(len(val_dataset))[:min_val_samples])
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True, **kwargs)
    
    return train_loader, val_loader, test_loader

async def train_batch(model, data, target, optimizer, criterion, device):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():  # Add no_grad for efficiency
        _, predicted = output.max(1)
        total = target.size(0)
        correct = predicted.eq(target).sum().item()
    
    return loss.item(), (100. * correct / total)

async def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    try:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item() * target.size(0)
                pred = output.max(1)[1]
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                del output, pred
        
        if total == 0:  # Add safety check
            return 0.0, 0.0
            
        return val_loss / total, 100. * correct / total
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return 0.0, 0.0  # Return default values in case of error

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

def get_random_test_samples(test_loader, num_samples=10, return_original=False):
    # Get random indices from the entire dataset
    dataset_size = len(test_loader.dataset)
    random_indices = torch.randperm(dataset_size)[:num_samples]
    
    if return_original:
        original_images = []
        labels = []
        transformed_images = []
        
        for idx in random_indices:
            # Get original image and label directly from the dataset
            original_img = test_loader.dataset.data[idx]
            label = test_loader.dataset.targets[idx]
            
            # Convert tensor to numpy array for original image
            if isinstance(original_img, torch.Tensor):
                original_img = original_img.numpy()
            
            # Get transformed image
            transformed_img = test_loader.dataset.transform(PIL.Image.fromarray(original_img))
            
            # Store the data
            original_images.append(original_img)
            labels.append(label.item() if isinstance(label, torch.Tensor) else label)
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images)
        labels = torch.tensor(labels)
        
        return transformed_images, labels, original_images
    
    # For non-original case
    images = []
    labels = []
    for idx in random_indices:
        img = test_loader.dataset.data[idx]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        label = test_loader.dataset.targets[idx]
        transformed_img = test_loader.dataset.transform(PIL.Image.fromarray(img))
        images.append(transformed_img)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    return torch.stack(images), torch.tensor(labels)

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

def get_model_summary(model):
    """Get model details including layers, parameters, and total trainable parameters"""
    summary = []
    total_params = 0
    trainable_params = 0
    
    # Get layer details
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            for sublayer_name, sublayer in layer.named_children():
                params = sum(p.numel() for p in sublayer.parameters())
                trainable = sum(p.numel() for p in sublayer.parameters() if p.requires_grad)
                summary.append({
                    'layer': f"{name}.{sublayer_name}",
                    'type': sublayer.__class__.__name__,
                    'params': params,
                    'trainable_params': trainable,
                    'shape': tuple(sublayer.weight.shape) if hasattr(sublayer, 'weight') else None
                })
                total_params += params
                trainable_params += trainable
        else:
            params = sum(p.numel() for p in layer.parameters())
            trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            summary.append({
                'layer': name,
                'type': layer.__class__.__name__,
                'params': params,
                'trainable_params': trainable,
                'shape': tuple(layer.weight.shape) if hasattr(layer, 'weight') else None
            })
            total_params += params
            trainable_params += trainable
    
    return {
        'layers': summary,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
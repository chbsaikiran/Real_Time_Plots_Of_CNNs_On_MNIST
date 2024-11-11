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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

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

def create_plot_data(losses, accuracies, model_name, progress, epoch, total_epochs, batch, total_batches, current_loss, current_acc):
    return {
        "model": model_name,
        f"progress{model_name[-1]}": progress,
        "losses": losses,
        "accuracies": accuracies,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "current_loss": current_loss,
        "current_acc": current_acc
    }

def get_random_test_samples(test_loader, num_samples=10):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    indices = torch.randperm(len(images))[:num_samples]
    return images[indices], labels[indices]
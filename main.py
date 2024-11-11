from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import socketio
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_models import CNNModel
from train_utils import get_data_loaders, train_batch, create_plot_data, get_random_test_samples, calculate_confusion_matrix, calculate_metrics
import json
import asyncio

# Create FastAPI and SocketIO apps
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Declare global variables for models
model1 = None
model2 = None

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@app.get("/train")
async def train_models(
    model1_conv1: int = Query(...),
    model1_conv2: int = Query(...),
    model1_conv3: int = Query(...),
    model1_optimizer: str = Query(...),
    model1_batch_size: int = Query(...),
    model1_epochs: int = Query(...),
    model2_conv1: int = Query(...),
    model2_conv2: int = Query(...),
    model2_conv3: int = Query(...),
    model2_optimizer: str = Query(...),
    model2_batch_size: int = Query(...),
    model2_epochs: int = Query(...),
):
    async def train_model(model, optimizer, criterion, train_loader, epochs, model_name):
        history = {"loss": [], "acc": [], "batch_count": 0}
        total_batches = len(train_loader) * epochs
        current_loss = 0
        current_acc = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    loss, acc = await train_batch(model, data, target, optimizer, criterion, device)
                    epoch_loss += loss
                    epoch_acc += acc
                    batch_count += 1
                    history["batch_count"] += 1
                    current_loss = epoch_loss / batch_count
                    current_acc = epoch_acc / batch_count
                    
                    # Update every 100 batches or at the end of epoch
                    if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                        history["loss"].append(current_loss)
                        history["acc"].append(current_acc)
                        progress = min(100, (history["batch_count"] * 100) // total_batches)
                        
                        # Create plot data
                        plot_data = create_plot_data(
                            history["loss"], 
                            history["acc"],
                            model_name, 
                            progress, 
                            epoch + 1, 
                            epochs,
                            batch_idx + 1, 
                            len(train_loader),
                            current_loss,
                            current_acc
                        )
                        
                        # Emit plot data through WebSocket
                        await sio.emit('plot_update', plot_data)
                        
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error in batch training: {str(e)}")
                    continue
            
            # Ensure we send an update at the end of each epoch
            if history["batch_count"] % 100 != 0:
                history["loss"].append(current_loss)
                history["acc"].append(current_acc)
                progress = min(100, (history["batch_count"] * 100) // total_batches)
                plot_data = create_plot_data(
                    history["loss"], 
                    history["acc"],
                    model_name, 
                    progress, 
                    epoch + 1, 
                    epochs,
                    len(train_loader), 
                    len(train_loader),
                    current_loss,
                    current_acc
                )
                await sio.emit('plot_update', plot_data)

    async def start_training():
        global model1, model2
        try:
            # Initialize models
            model1 = CNNModel(model1_conv1, model1_conv2, model1_conv3).to(device)
            model2 = CNNModel(model2_conv1, model2_conv2, model2_conv3).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer1 = (optim.Adam(model1.parameters()) if model1_optimizer == "adam" 
                         else optim.SGD(model1.parameters(), lr=0.01))
            optimizer2 = (optim.Adam(model2.parameters()) if model2_optimizer == "adam" 
                         else optim.SGD(model2.parameters(), lr=0.01))
            
            train_loader1, test_loader1 = get_data_loaders(model1_batch_size)
            train_loader2, test_loader2 = get_data_loaders(model2_batch_size)

            # Train both models concurrently
            await asyncio.gather(
                train_model(model1, optimizer1, criterion, train_loader1, model1_epochs, "model1"),
                train_model(model2, optimizer2, criterion, train_loader2, model2_epochs, "model2")
            )

            await sio.emit('training_complete', {
                "status": "complete"
            })

        except Exception as e:
            print(f"Error in training: {str(e)}")
            await sio.emit('training_error', {"error": str(e)})

    # Start training in background
    asyncio.create_task(start_training())
    return JSONResponse({"status": "Training started"})

@app.get("/get_test_results")
async def get_test_results():
    try:
        # Get data loaders
        _, test_loader = get_data_loaders(batch_size=100)
        
        # Get random test samples
        test_images, test_labels = get_random_test_samples(test_loader)
        test_images = test_images.to(device)
        
        # Get predictions from both models
        with torch.no_grad():
            pred1 = model1(test_images).max(1)[1]
            pred2 = model2(test_images).max(1)[1]
        
        return {
            "true_labels": test_labels.tolist(),
            "model1_preds": pred1.tolist(),
            "model2_preds": pred2.tolist()
        }
    except Exception as e:
        print(f"Error getting test results: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_model_metrics")
async def get_model_metrics():
    try:
        # Get data loaders
        _, test_loader = get_data_loaders(batch_size=100)
        
        # Initialize metrics
        all_labels = []
        all_preds1 = []
        all_preds2 = []
        
        # Get predictions for entire test set
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                pred1 = model1(images).max(1)[1]
                pred2 = model2(images).max(1)[1]
                
                all_labels.extend(labels.tolist())
                all_preds1.extend(pred1.tolist())
                all_preds2.extend(pred2.tolist())
        
        # Calculate confusion matrices
        conf_matrix1 = calculate_confusion_matrix(all_labels, all_preds1)
        conf_matrix2 = calculate_confusion_matrix(all_labels, all_preds2)
        
        # Calculate metrics for both models
        metrics1 = calculate_metrics(conf_matrix1)
        metrics2 = calculate_metrics(conf_matrix2)
        
        return {
            "model1": {
                "confusion_matrix": conf_matrix1.tolist(),
                **metrics1
            },
            "model2": {
                "confusion_matrix": conf_matrix2.tolist(),
                **metrics2
            }
        }
    except Exception as e:
        print(f"Error getting model metrics: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

app = socket_app
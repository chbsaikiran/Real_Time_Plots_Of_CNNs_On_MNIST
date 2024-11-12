from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import socketio
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_models import CNNModel
from train_utils import get_data_loaders, train_batch, create_plot_data, get_random_test_samples, calculate_confusion_matrix, calculate_metrics, evaluate_model
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
train_loader1 = None
val_loader1 = None
test_loader1 = None
train_loader2 = None
val_loader2 = None
test_loader2 = None

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
    async def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, model_name):
        history = {
            "train_loss": [], "train_acc": [], 
            "val_loss": [], "val_acc": [], 
            "batch_count": 0
        }
        total_batches = len(train_loader) * epochs
        running_loss = 0
        running_acc = 0
        total_samples = 0
        
        # Get initial validation metrics
        val_loss, val_acc = await evaluate_model(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    loss, acc = await train_batch(model, data, target, optimizer, criterion, device)
                    batch_size = target.size(0)
                    
                    # Update running statistics with weighted averages
                    running_loss = (running_loss * total_samples + loss * batch_size) / (total_samples + batch_size)
                    running_acc = (running_acc * total_samples + acc * batch_size) / (total_samples + batch_size)
                    total_samples += batch_size
                    history["batch_count"] += 1
                    
                    # Update training metrics every 10 batches
                    if batch_idx % 10 == 0:
                        history["train_loss"].append(running_loss)
                        history["train_acc"].append(running_acc)
                        progress = min(100, (history["batch_count"] * 100) // total_batches)
                        
                        # Create plot data
                        plot_data = create_plot_data(
                            history["train_loss"], 
                            history["train_acc"],
                            history["val_loss"],
                            history["val_acc"],
                            model_name, 
                            progress, 
                            epoch + 1, 
                            epochs,
                            batch_idx + 1, 
                            len(train_loader),
                            running_loss,
                            running_acc,
                            history["val_loss"][-1],
                            history["val_acc"][-1]
                        )
                        
                        # Emit plot data through WebSocket
                        await sio.emit('plot_update', plot_data)
                    
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error in batch training: {str(e)}")
                    continue
            
            # Evaluate on validation set at the end of each epoch
            val_loss, val_acc = await evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Send final update for this epoch
            progress = min(100, (history["batch_count"] * 100) // total_batches)
            plot_data = create_plot_data(
                history["train_loss"], 
                history["train_acc"],
                history["val_loss"],
                history["val_acc"],
                model_name, 
                progress, 
                epoch + 1, 
                epochs,
                len(train_loader), 
                len(train_loader),
                running_loss,
                running_acc,
                val_loss,
                val_acc
            )
            await sio.emit('plot_update', plot_data)

    async def start_training():
        global model1, model2, train_loader1, val_loader1, test_loader1, train_loader2, val_loader2, test_loader2
        try:
            # Initialize models
            model1 = CNNModel(model1_conv1, model1_conv2, model1_conv3).to(device)
            model2 = CNNModel(model2_conv1, model2_conv2, model2_conv3).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer1 = (optim.Adam(model1.parameters()) if model1_optimizer == "adam" 
                         else optim.SGD(model1.parameters(), lr=0.01))
            optimizer2 = (optim.Adam(model2.parameters()) if model2_optimizer == "adam" 
                         else optim.SGD(model2.parameters(), lr=0.01))
            
            # Store loaders globally
            train_loader1, val_loader1, test_loader1 = get_data_loaders(model1_batch_size)
            train_loader2, val_loader2, test_loader2 = get_data_loaders(model2_batch_size)

            # Train both models concurrently
            await asyncio.gather(
                train_model(model1, optimizer1, criterion, train_loader1, val_loader1, model1_epochs, "model1"),
                train_model(model2, optimizer2, criterion, train_loader2, val_loader2, model2_epochs, "model2")
            )

            # Emit training complete event
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
async def get_model_metrics(data_type: str = Query(...)):
    try:
        if model1 is None or model2 is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Models not trained yet. Please train the models first."}
            )

        # Choose appropriate loader based on data_type
        if data_type == 'train':
            loader1, loader2 = train_loader1, train_loader2
        elif data_type == 'test':
            loader1, loader2 = test_loader1, test_loader2
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid data_type. Use 'train' or 'test'."}
            )
        
        # Initialize metrics
        all_labels1, all_preds1 = [], []
        all_labels2, all_preds2 = [], []
        
        # Get predictions for both models
        with torch.no_grad():
            for images, labels in loader1:
                images = images.to(device)
                pred1 = model1(images).max(1)[1]
                all_labels1.extend(labels.tolist())
                all_preds1.extend(pred1.tolist())
            
            for images, labels in loader2:
                images = images.to(device)
                pred2 = model2(images).max(1)[1]
                all_labels2.extend(labels.tolist())
                all_preds2.extend(pred2.tolist())
        
        # Calculate confusion matrices
        conf_matrix1 = calculate_confusion_matrix(all_labels1, all_preds1)
        conf_matrix2 = calculate_confusion_matrix(all_labels2, all_preds2)
        
        # Calculate metrics for both models
        metrics1 = calculate_metrics(conf_matrix1)
        metrics2 = calculate_metrics(conf_matrix2)
        
        return {
            "model1": {
                "confusion_matrix": conf_matrix1.tolist(),
                **metrics1,
                "data_type": data_type
            },
            "model2": {
                "confusion_matrix": conf_matrix2.tolist(),
                **metrics2,
                "data_type": data_type
            }
        }
    except Exception as e:
        print(f"Error getting model metrics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error calculating metrics: {str(e)}"}
        )

app = socket_app
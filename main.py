from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import socketio
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_models import CNNModel
from train_utils import get_data_loaders, train_batch, create_plot_data, get_random_test_samples, calculate_confusion_matrix, calculate_metrics, evaluate_model, get_model_summary
import json
import asyncio
import threading
from queue import Queue
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import io
import base64

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

# Move ModelTrainer class outside the endpoint
class ModelTrainer:
    def __init__(self, device):
        self.device = device
        self.completed_models = 0

    def train_model_thread(self, model_params, queue):
        try:
            # Unpack parameters
            conv1, conv2, conv3, opt_type, batch_size, epochs, model_num = model_params
            
            # Initialize model and store globally
            if model_num == 1:
                global model1, train_loader1, val_loader1, test_loader1
                model1 = CNNModel(conv1, conv2, conv3).to(self.device)
                model = model1
                train_loader, val_loader, test_loader = get_data_loaders(batch_size)
                train_loader1, val_loader1, test_loader1 = train_loader, val_loader, test_loader
            else:
                global model2, train_loader2, val_loader2, test_loader2
                model2 = CNNModel(conv1, conv2, conv3).to(self.device)
                model = model2
                train_loader, val_loader, test_loader = get_data_loaders(batch_size)
                train_loader2, val_loader2, test_loader2 = train_loader, val_loader, test_loader

            criterion = nn.CrossEntropyLoss()
            optimizer = (optim.Adam(model.parameters()) if opt_type == "adam" 
                       else optim.SGD(model.parameters(), lr=0.01))
            
            # Add learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
            
            # Get and send model summary before training starts
            model_summary = get_model_summary(model)
            queue.put({
                "model": f"model{model_num}",
                "type": "model_summary",
                "summary": model_summary,
                "config": {
                    "conv1": conv1,
                    "conv2": conv2,
                    "conv3": conv3,
                    "optimizer": opt_type,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "initial_lr": optimizer.param_groups[0]['lr']
                }
            })
            
            # Initialize training variables
            total_batches = len(train_loader) * epochs
            running_loss = 0
            running_acc = 0
            total_samples = 0
            history = {
                "train_loss": [], 
                "train_acc": [],
                "val_loss": [], 
                "val_acc": [],
                "batch_count": 0,
                "learning_rates": []
            }
            
            # Initial validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.max(1)[1]
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    pred = output.max(1)[1]
                    acc = 100. * pred.eq(target).sum().item() / len(target)
                    
                    batch_size = target.size(0)
                    running_loss = (running_loss * total_samples + loss.item() * batch_size) / (total_samples + batch_size)
                    running_acc = (running_acc * total_samples + acc * batch_size) / (total_samples + batch_size)
                    total_samples += batch_size
                    history["batch_count"] += 1
                    
                    if batch_idx % 250 == 0 or batch_idx == len(train_loader) - 1:
                        history["train_loss"].append(running_loss)
                        history["train_acc"].append(running_acc)
                        progress = min(100, (history["batch_count"] * 100) // total_batches)
                        
                        queue.put({
                            "model": f"model{model_num}",
                            f"progress{model_num}": progress,
                            "train_losses": history["train_loss"],
                            "train_accuracies": history["train_acc"],
                            "val_losses": history["val_loss"],
                            "val_accuracies": history["val_acc"],
                            "epoch": epoch + 1,
                            "total_epochs": epochs,
                            "batch": batch_idx + 1,
                            "total_batches": len(train_loader),
                            "current_loss": running_loss,
                            "current_acc": running_acc,
                            "current_val_loss": history["val_loss"][-1],
                            "current_val_acc": history["val_acc"][-1],
                            "current_lr": optimizer.param_groups[0]['lr']
                        })
                
                # Validation at end of epoch
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        pred = output.max(1)[1]
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
                
                val_loss /= len(val_loader)
                val_acc = 100. * correct / total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Update learning rate
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                history["learning_rates"].append(current_lr)

                # Send final update for this epoch
                queue.put({
                    "model": f"model{model_num}",
                    f"progress{model_num}": min(100, ((epoch + 1) * 100) // epochs),
                    "train_losses": history["train_loss"],
                    "train_accuracies": history["train_acc"],
                    "val_losses": history["val_loss"],
                    "val_accuracies": history["val_acc"],
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "batch": len(train_loader),
                    "total_batches": len(train_loader),
                    "current_loss": running_loss,
                    "current_acc": running_acc,
                    "current_val_loss": val_loss,
                    "current_val_acc": val_acc,
                    "current_lr": current_lr
                })

            # Signal completion
            queue.put({
                "model": f"model{model_num}", 
                "status": "complete"
            })
            
        except Exception as e:
            queue.put({"model": f"model{model_num}", "error": str(e)})

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
    async def process_queue(queue):
        completed_models = 0
        while completed_models < 2:
            try:
                # Check queue more frequently
                for _ in range(10):
                    try:
                        data = queue.get_nowait()
                        if "error" in data:
                            await sio.emit('training_error', data)
                            return
                        elif "status" in data and data["status"] == "complete":
                            completed_models += 1
                            await sio.emit('training_complete', data)
                            if completed_models == 2:
                                await sio.emit('training_complete', {"status": "all_complete"})
                        else:
                            await sio.emit('plot_update', data)
                    except:
                        continue
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in process_queue: {str(e)}")
                await asyncio.sleep(0.01)

    async def start_training():
        try:
            queue = Queue()
            trainer = ModelTrainer(device)
            
            # Create thread parameters
            model1_params = (model1_conv1, model1_conv2, model1_conv3, 
                           model1_optimizer, model1_batch_size, model1_epochs, 1)
            model2_params = (model2_conv1, model2_conv2, model2_conv3, 
                           model2_optimizer, model2_batch_size, model2_epochs, 2)
            
            # Start threads
            t1 = threading.Thread(target=trainer.train_model_thread, 
                                args=(model1_params, queue))
            t2 = threading.Thread(target=trainer.train_model_thread, 
                                args=(model2_params, queue))
            
            t1.start()
            t2.start()
            
            # Process queue in background
            await process_queue(queue)
            
            # Wait for threads to complete
            t1.join()
            t2.join()

        except Exception as e:
            print(f"Error in training: {str(e)}")
            await sio.emit('training_error', {"error": str(e)})
        finally:
            # Cleanup
            if 'queue' in locals():
                while not queue.empty():
                    _ = queue.get()

    # Start training in background
    asyncio.create_task(start_training())
    return JSONResponse({"status": "Training started"})

@app.get("/get_test_results")
async def get_test_results():
    try:
        if model1 is None or model2 is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Models not trained yet. Please train the models first."}
            )
        
        # Get random test samples with original images
        try:
            test_images, test_labels, original_images = get_random_test_samples(test_loader1, return_original=True)
        except Exception as e:
            print(f"Error getting test samples: {str(e)}")
            raise
        
        # Convert original images to displayable format
        images_data = []
        try:
            for img in original_images:
                # Convert to bytes (original images are already in 0-255 range)
                img_bytes = io.BytesIO()
                import PIL.Image
                PIL.Image.fromarray(img).save(img_bytes, format='PNG')
                img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                images_data.append(img_base64)
        except Exception as e:
            print(f"Error converting images: {str(e)}")
            raise
        
        # Get predictions using transformed images
        try:
            test_images = test_images.to(device)
            with torch.no_grad():
                model1.eval()
                model2.eval()
                pred1 = model1(test_images).max(1)[1]
                pred2 = model2(test_images).max(1)[1]
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            raise
        
        return {
            "images": images_data,
            "true_labels": test_labels.tolist(),
            "model1_preds": pred1.tolist(),
            "model2_preds": pred2.tolist()
        }
    except Exception as e:
        print(f"Error in get_test_results: {str(e)}")
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
        
        # Use larger batch size for faster processing
        batch_size = 1000
        
        # Get predictions for both models
        with torch.no_grad():
            model1.eval()
            model2.eval()
            
            # Process only a subset of data for faster metrics
            max_samples = 5000  # Limit number of samples
            samples_processed = 0
            
            for images, labels in loader1:
                if samples_processed >= max_samples:
                    break
                    
                images = images.to(device)
                pred1 = model1(images).max(1)[1]
                all_labels1.extend(labels.tolist())
                all_preds1.extend(pred1.tolist())
                samples_processed += len(images)
            
            samples_processed = 0
            for images, labels in loader2:
                if samples_processed >= max_samples:
                    break
                    
                images = images.to(device)
                pred2 = model2(images).max(1)[1]
                all_labels2.extend(labels.tolist())
                all_preds2.extend(pred2.tolist())
                samples_processed += len(images)
        
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
                "data_type": data_type,
                "samples_evaluated": len(all_labels1)
            },
            "model2": {
                "confusion_matrix": conf_matrix2.tolist(),
                **metrics2,
                "data_type": data_type,
                "samples_evaluated": len(all_labels2)
            }
        }
    except Exception as e:
        print(f"Error getting model metrics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error calculating metrics: {str(e)}"}
        )

app = socket_app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
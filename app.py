import asyncio
import uuid
import io
import os
import base64
import json
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Try to import torch; gracefully degrade if unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - using simulated detection")

app = FastAPI(title="Sleep-Integrated Training Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture
class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=1):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(100)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)

# Model loading functionality
MODEL_PATH = "rem_model.pth"
loaded_model = None

if TORCH_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            model = CNNBiLSTM()
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            loaded_model = model
            print("✅ REM model loaded successfully from", MODEL_PATH)
        else:
            print("⚠️  Model file not found at:", MODEL_PATH)
            print("   Using simulated REM detection")
    except Exception as e:
        print("❌ Failed to load model:", e)
        print("   Using simulated REM detection")
        loaded_model = None
else:
    print("🔶 PyTorch not available - using simulated REM detection")

# Simple in-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        self.active_connections.pop(job_id, None)

    async def send_message(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json(message)
            except:
                self.disconnect(job_id)

manager = ConnectionManager()

async def process_eeg_data_with_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process EEG data using the loaded PyTorch model
    """
    if not TORCH_AVAILABLE or loaded_model is None:
        return await process_eeg_data_simulated(df)

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {"prediction": "unknown", "confidence": 0.0, "reason": "No numeric data"}

        series = df[numeric_cols[0]].fillna(0).astype(float).values

        # Preprocess data for model
        if len(series) > 1000:
            series = series[:1000]
        elif len(series) < 1000:
            series = np.pad(series, (0, 1000 - len(series)), 'constant')

        tensor_data = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

        # Model prediction
        with torch.no_grad():
            output = loaded_model(tensor_data)
            confidence = output.item()

        prediction = "REM" if confidence > 0.5 else "Non-REM"

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "model_used": "CNNBiLSTM",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Model inference error: {e}")
        return await process_eeg_data_simulated(df)

async def process_eeg_data_simulated(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fallback EEG data processing when model is not available
    """
    eeg_columns = [col for col in df.columns if 'EEG' in col and '_t' in col]

    if not eeg_columns:
        eeg_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not eeg_columns:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "reason": "No numeric data columns found",
            "model_used": "simulated"
        }

    results = []

    for channel in eeg_columns[:5]:
        try:
            series = df[channel].astype(float)
            variance = float(series.var())
            mean = float(series.mean())
            std_dev = float(series.std())

            is_rem_likely = (
                variance > 1e-12 and
                abs(mean) < 0.01 and
                len(series) > 5
            )

            confidence = min(0.95, max(0.1,
                (min(variance * 1e10, 1.0)) * 0.3 +
                (min(std_dev * 1e5, 1.0)) * 0.2 +
                0.4
            ))

            results.append({
                "channel": channel,
                "prediction": "REM" if is_rem_likely else "Non-REM",
                "confidence": float(confidence),
                "variance": variance,
                "mean": mean,
                "std_dev": std_dev
            })

        except Exception as e:
            results.append({
                "channel": channel,
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            })

    rem_count = sum(1 for r in results if r.get('prediction') == 'REM' and r.get('confidence', 0) > 0.5)
    total_valid = sum(1 for r in results if r.get('prediction') in ['REM', 'Non-REM'])

    if total_valid == 0:
        final_prediction = "unknown"
        final_confidence = 0.0
    else:
        rem_ratio = rem_count / total_valid
        final_prediction = "REM" if rem_ratio > 0.5 else "Non-REM"
        final_confidence = rem_ratio

    return {
        "prediction": final_prediction,
        "confidence": float(final_confidence),
        "channel_analysis": results,
        "rem_channels": rem_count,
        "total_channels": total_valid,
        "rem_ratio": rem_ratio if total_valid > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "model_used": "simulated"
    }

# Use model if available, otherwise use simulated
process_eeg_data = process_eeg_data_with_model if (TORCH_AVAILABLE and loaded_model is not None) else process_eeg_data_simulated

@app.post("/upload-csv")
async def upload_csv_base64(
    filename: str = Body(..., embed=True),
    content_base64: str = Body(..., embed=True)
):
    """
    Upload CSV via base64 encoding - ONLY endpoint for data upload
    """
    if not filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    job_id = str(uuid.uuid4())

    try:
        # Decode base64 content
        csv_content = base64.b64decode(content_base64)

        # Parse CSV
        df = pd.read_csv(io.BytesIO(csv_content))

        # Store job data
        jobs[job_id] = {
            "status": "uploaded",
            "filename": filename,
            "dataframe": df.to_dict('records'),
            "columns": list(df.columns),
            "shape": df.shape,
            "result": None,
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }

        return JSONResponse({
            "job_id": job_id,
            "filename": filename,
            "data_shape": df.shape,
            "columns": list(df.columns),
            "status": "uploaded",
            "model_available": loaded_model is not None
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "status": job["status"],
        "progress": job["progress"],
        "filename": job.get("filename", "unknown"),
        "created_at": job.get("created_at"),
        "model_available": loaded_model is not None
    }

    if job["result"]:
        response["result"] = job["result"]

    return response

@app.get("/model-status")
async def get_model_status():
    """Check if the ML model is loaded and available"""
    return {
        "torch_available": TORCH_AVAILABLE,
        "model_loaded": loaded_model is not None,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH) if TORCH_AVAILABLE else False
    }

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "total_jobs": len(jobs),
        "model_available": loaded_model is not None,
        "jobs": {
            job_id: {
                "status": job["status"],
                "filename": job.get("filename"),
                "progress": job["progress"],
                "created_at": job.get("created_at")
            }
            for job_id, job in jobs.items()
        }
    }

async def run_detection(job_id: str, websocket: WebSocket):
    job = jobs.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "job not found"})
        return

    try:
        # Update job status
        jobs[job_id]['status'] = 'processing'

        # Send model status
        model_status = "CNNBiLSTM Model" if loaded_model is not None else "Simulated Detection"
        await websocket.send_json({
            "type": "status",
            "event": "model_loaded",
            "detail": f"Using {model_status}",
            "progress": 10
        })

        # Simulate connection sequence
        connection_steps = [
            {"delay": 1.0, "event": "ECG connected", "detail": "ECG lead readings nominal"},
            {"delay": 1.5, "event": "EOG connected", "detail": "EOG channels online"},
            {"delay": 1.2, "event": "BP normal", "detail": "Blood pressure in normal range"},
            {"delay": 0.8, "event": "EEG calibrated", "detail": "All EEG channels calibrated"},
        ]

        # Send connection progress
        for i, step in enumerate(connection_steps, start=1):
            await asyncio.sleep(step['delay'])
            progress = 10 + int((i / (len(connection_steps) + 3)) * 60)
            jobs[job_id]['progress'] = progress

            await websocket.send_json({
                "type": "status",
                "event": step['event'],
                "detail": step['detail'],
                "progress": progress
            })

        # Data analysis phase
        await asyncio.sleep(1.0)
        jobs[job_id]['progress'] = 80
        await websocket.send_json({
            "type": "status",
            "event": "analyzing_eeg",
            "detail": "Processing EEG data for REM detection",
            "progress": 80
        })

        # Convert stored records back to DataFrame
        df = pd.DataFrame(job["dataframe"])

        # Perform REM detection
        await asyncio.sleep(2.0)
        result = await process_eeg_data(df)

        # Update job status
        jobs[job_id]['result'] = result
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100

        # Send final result
        await websocket.send_json({
            "type": "result",
            "result": result,
            "progress": 100
        })

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['result'] = {"error": str(e)}
        await websocket.send_json({
            "type": "error",
            "message": f"Analysis failed: {str(e)}"
        })


# new function
async def run_detection(job_id: str, websocket: WebSocket):
    job = jobs.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "job not found"})
        return

    try:
        # Update job status
        jobs[job_id]['status'] = 'processing'

        # Send model status
        model_status = "CNNBiLSTM Model" if loaded_model is not None else "Simulated Detection"
        await websocket.send_json({
            "type": "status",
            "event": "model_loaded",
            "detail": f"Using {model_status}",
            "progress": 10
        })

        # Simulate connection sequence
        connection_steps = [
            {"delay": 1.0, "event": "ECG connected", "detail": "ECG lead readings nominal"},
            {"delay": 1.5, "event": "EOG connected", "detail": "EOG channels online"},
            {"delay": 1.2, "event": "BP normal", "detail": "Blood pressure in normal range"},
            {"delay": 0.8, "event": "EEG calibrated", "detail": "All EEG channels calibrated"},
        ]

        for i, step in enumerate(connection_steps, start=1):
            await asyncio.sleep(step['delay'])
            progress = 10 + int((i / (len(connection_steps) + 3)) * 60)
            jobs[job_id]['progress'] = progress

            await websocket.send_json({
                "type": "status",
                "event": step['event'],
                "detail": step['detail'],
                "progress": progress
            })

        # Data analysis phase
        await asyncio.sleep(1.0)
        jobs[job_id]['progress'] = 80
        await websocket.send_json({
            "type": "status",
            "event": "analyzing_eeg",
            "detail": "Processing EEG data for REM detection",
            "progress": 80
        })

        # Convert stored records back to DataFrame
        df = pd.DataFrame(job["dataframe"])

        # Perform REM detection
        await asyncio.sleep(2.0)
        result = await process_eeg_data(df)

        # ✅ If REM detected, instantly push event
        if result.get("prediction") == "REM":
            await websocket.send_json({
                "type": "event",
                "event": "REM_detected",
                "detail": "REM sleep detected, triggering learning module",
                "timestamp": datetime.now().isoformat()
            })

        # Update job status
        jobs[job_id]['result'] = result
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100

        # Send final result
        await websocket.send_json({
            "type": "result",
            "result": result,
            "progress": 100
        })

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['result'] = {"error": str(e)}
        await websocket.send_json({
            "type": "error",
            "message": f"Analysis failed: {str(e)}"
        })




@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        await run_detection(job_id, websocket)
    except WebSocketDisconnect:
        manager.disconnect(job_id)
        print(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        manager.disconnect(job_id)

@app.get("/")
async def root():
    return {
        "message": "Sleep Training Backend API",
        "status": "running",
        "model_available": loaded_model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(jobs),
        "active_connections": len(manager.active_connections),
        "model_available": loaded_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

import asyncio
import uuid
import io
import os
import base64
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Try to import torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - using simulated detection")

app = FastAPI(title="REM Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Model definition
# --------------------------
class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(100)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
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

# Load model if exists
MODEL_PATH = "rem_model.pth"
loaded_model = None
if TORCH_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
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
        print("✅ REM model loaded")
    except Exception as e:
        print("❌ Failed to load model:", e)
        loaded_model = None
else:
    print("🔶 Using simulated detection")

# --------------------------
# In-memory jobs store
# --------------------------
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

# --------------------------
# EEG Processing
# --------------------------
async def process_eeg_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Try model prediction first, fallback to simulated.
    """
    # Trim to max 2000 rows for memory safety
    df = df.head(2000)

    if TORCH_AVAILABLE and loaded_model is not None:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found")

            series = df[numeric_cols[0]].fillna(0).astype(float).values
            series = series[:1000] if len(series) > 1000 else np.pad(series, (0, 1000 - len(series)), 'constant')
            tensor_data = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                confidence = float(loaded_model(tensor_data).item())

            prediction = "REM" if confidence > 0.5 else "Non-REM"

            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_used": "CNNBiLSTM",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print("Model inference failed, fallback to simulated:", e)

    # Fallback simulated
    return await process_eeg_data_simulated(df)

async def process_eeg_data_simulated(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"prediction": "unknown", "confidence": 0.0, "model_used": "simulated"}

    series = df[numeric_cols[0]].fillna(0).astype(float)
    var = float(series.var())
    mean = float(series.mean())
    std = float(series.std())

    is_rem = (var > 1e-12 and abs(mean) < 0.01)
    confidence = 0.7 if is_rem else 0.3

    return {
        "prediction": "REM" if is_rem else "Non-REM",
        "confidence": confidence,
        "model_used": "simulated",
        "variance": var,
        "mean": mean,
        "std_dev": std,
        "timestamp": datetime.now().isoformat()
    }

# --------------------------
# Upload CSV endpoint
# --------------------------
@app.post("/upload-csv")
async def upload_csv_base64(
    filename: str = Body(..., embed=True),
    content_base64: str = Body(..., embed=True)
):
    if not filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV allowed")

    job_id = str(uuid.uuid4())

    try:
        csv_content = base64.b64decode(content_base64)
        df = pd.read_csv(io.BytesIO(csv_content))

        jobs[job_id] = {
            "status": "uploaded",
            "filename": filename,
            "dataframe": df.head(2000).to_dict('records'),  # memory-safe
            "progress": 0,
            "result": None,
            "created_at": datetime.now().isoformat()
        }
        return {"job_id": job_id, "status": "uploaded", "columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed: {e}")

# --------------------------
# WebSocket processing
# --------------------------
async def run_detection(job_id: str, websocket: WebSocket):
    job = jobs.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        return

    jobs[job_id]['status'] = 'processing'
    await websocket.send_json({"type": "status", "event": "processing_started", "progress": 10})

    try:
        df = pd.DataFrame(job["dataframe"])
        result = await process_eeg_data(df)

        # Store result & free memory
        jobs[job_id]['result'] = result
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['dataframe'] = None

        await websocket.send_json({"type": "result", "result": result, "progress": 100})

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['result'] = {"error": str(e)}
        await websocket.send_json({"type": "error", "message": str(e)})

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        await run_detection(job_id, websocket)
    except WebSocketDisconnect:
        manager.disconnect(job_id)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        manager.disconnect(job_id)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_jobs": len(jobs),
        "active_connections": len(manager.active_connections),
        "model_available": loaded_model is not None
    }

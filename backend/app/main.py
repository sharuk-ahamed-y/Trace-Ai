"""
backend/app/main.py — FastAPI Web Application Server Entrypoint
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import BASE_DIR, DATA_DIR, UPLOAD_DIR, REPORT_DIR, VISUALIZATION_DIR, DATASET_DIR
from backend.app.database.db import init_db
from backend.app.services.model_manager import get_model_manager
from backend.app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("==================================================")
    print(" Starting TRACEAI — Image Provenance Server ")
    print("==================================================")
    init_db()
    print("[DB] SQLite database initialized.")
    
    # Pre-initialize model manager in background / startup
    try:
        manager = get_model_manager()
        print(f"[Model] Engine ready on {manager.device}.")
    except Exception as e:
        print(f"[Model] Initialization warning: {e}")

    yield

    print("[Server] Shutting down TRACEAI backend.")

app = FastAPI(
    title="TRACEAI — Image Provenance Analysis & Graph Reconstruction",
    description="Full-stack AI Web Platform for Image Provenance Analysis, MST Graph Reconstruction, and XAI Explanations.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React Frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API router
app.include_router(api_router, prefix="/api")

# Mount Static File Routes
app.mount("/data/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/data/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")
app.mount("/data/visualizations", StaticFiles(directory=str(VISUALIZATION_DIR)), name="visualizations")

dataset_images_dir = (DATASET_DIR / "images") if (DATASET_DIR / "images").exists() else ((BASE_DIR / "data" / "provenance_dataset_v2" / "images") if (BASE_DIR / "data" / "provenance_dataset_v2" / "images").exists() else None)
if dataset_images_dir is not None:
    app.mount("/data/dataset-images", StaticFiles(directory=str(dataset_images_dir)), name="dataset_images")

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "app": "TRACEAI Provenance Platform API",
            "status": "ONLINE",
            "documentation": "/docs",
            "health": "/api/health"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes.evaluation import router as evaluation_router
from backend.api.routes.health import router as health_router
from backend.api.routes.synthesis import router as synthesis_router
from backend.config import OUTPUT_DIRECTORY

app = FastAPI(title="Controllable Zero-Shot Voice Cloning", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(synthesis_router)
app.include_router(evaluation_router)
app.mount("/audio", StaticFiles(directory=OUTPUT_DIRECTORY), name="audio")


@app.get("/")
def root():
    return {"message": "Controllable Zero-Shot Voice Cloning API"}

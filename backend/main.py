from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.deps import cleanup_old_outputs
from backend.api.routes.evaluation import router as evaluation_router
from backend.api.routes.health import router as health_router
from backend.api.routes.prosody import router as prosody_router
from backend.api.routes.synthesis import router as synthesis_router
from backend.config import CORS_ORIGINS, OUTPUT_DIRECTORY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(_: FastAPI):
    cleanup_old_outputs()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Controllable Zero-Shot Voice Cloning",
    version="1.0.0",
    description="Zero-shot voice cloning (pretrained F5-TTS) with an explicit prosody controller.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(synthesis_router)
app.include_router(prosody_router)
app.include_router(evaluation_router)
app.mount("/audio", StaticFiles(directory=OUTPUT_DIRECTORY), name="audio")


@app.get("/")
def root():
    return {"message": "Controllable Zero-Shot Voice Cloning API", "docs": "/docs"}

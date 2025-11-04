from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import numpy as np
from io import BytesIO

from core.analysis import analyze_wave

app = FastAPI(title="i-onetake-analyzer", version="1.0.0")

# CORS（GitHub Pages等の別オリジンから叩けるように）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じてドメインを限定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    profile: str
    f0_median_hz: float | None
    f0_std_hz: float | None
    vtl_cm: float | None
    vtl_cv: float | None
    vtl_n_keep: int
    height_cm: float | None
    alarms: list

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    profile: str = Form("auto"),
    known_height_cm: float | None = Form(None)
):
    data = await file.read()
    bio = BytesIO(data)
    y, sr = sf.read(bio, dtype="float32", always_2d=False)
    if y.ndim == 2:  # stereo -> mono
        y = np.mean(y, axis=1)
    result = analyze_wave(y, sr, profile=profile, known_height_cm=known_height_cm)
    return result

@app.get("/healthz")
def healthz():
    return {"ok": True}

# server/app.py 先頭付近
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ← これを追加（/opt/render/project/src をパスに）

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import wave
from io import BytesIO

from core.analysis import analyze_wave


app = FastAPI(title="i-onetake-analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 公開時は "https://t-mimbu.github.io" に絞るのが安全
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

sem = anyio.Semaphore(1)  # ← 同時1リクエスト（任意だが安定する）

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    profile: str = Form("auto"),
    known_height_cm: float | None = Form(None)
):
    async with sem:
        data = await file.read()

        # WAV(PCM16, mono, 48kHz) を標準ライブラリで読む
        bio = BytesIO(data)
        with wave.open(bio, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.getnframes()
            raw = wf.readframes(frames)

        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM is supported")

        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch == 2:  # stereo→mono
            y = y.reshape(-1, 2).mean(axis=1)

        # 解析
        result = analyze_wave(y, sr, profile=profile, known_height_cm=known_height_cm)
        return result   # ← 重複してた片方を残すだけ

@app.get("/healthz")
def healthz():
    return {"ok": True}

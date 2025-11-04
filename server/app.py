# server/app.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ../ を import path に追加

import anyio

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import wave
from io import BytesIO
import traceback

from core.analysis import analyze_wave


app = FastAPI(title="i-onetake-analyzer", version="1.0.0")

# 公開時は allow_origins を必要なドメインに絞るのが安全（例：GitHub Pages のURL）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 例: ["https://t-mimbu.github.io"]
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

# 同時実行を 1 リクエストに制限（Render の無料インスタンスで安定動作）
sem = anyio.Semaphore(1)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    profile: str = Form("auto"),
    known_height_cm: float | None = Form(None)
):
    async with sem:
        try:
            data = await file.read()

            # WAV を標準ライブラリで読む
            bio = BytesIO(data)
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.getnframes()
                raw = wf.readframes(frames)

            # 16-bit PCM だけ受け付ける（それ以外は 400）
            if sampwidth != 2:
                raise HTTPException(status_code=400, detail="Only 16-bit PCM WAV is supported.")

            # int16 → float32 へ正規化、ステレオは平均化
            y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if ch == 2:
                y = y.reshape(-1, 2).mean(axis=1)

            # 解析
            result = analyze_wave(y, sr, profile=profile, known_height_cm=known_height_cm)
            return result

        except HTTPException:
            # 上で明示的に投げた 400 などはそのまま返す
            raise
        except wave.Error as e:
            # WAV ヘッダが壊れている/圧縮など
            raise HTTPException(status_code=400, detail=f"WAV parse error: {e}")
        except Exception:
            # 想定外は 500。ログには詳細を残す
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal error")

@app.get("/healthz")
def healthz():
    return {"ok": True}

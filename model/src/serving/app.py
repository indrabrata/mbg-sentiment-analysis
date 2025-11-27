from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib, os, json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

MODEL_JOBLIB = "results_indobertweet/model_pipeline.joblib"
EVAL_DIR = "results_indobertweet/eval"
HF_MODEL_DIR = "results_indobertweet/best_model"

# ===== Label mapping =====
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

app = FastAPI(title="Sentiment API (IndoBERTweet)")

# load joblib pipeline at startup
@app.on_event("startup")
def load_model():
    global MODEL_PIPELINE
    if not os.path.exists(MODEL_JOBLIB):
        raise RuntimeError(f"Model joblib tidak ditemukan di {MODEL_JOBLIB}. Jalankan training dulu.")
    MODEL_PIPELINE = joblib.load(MODEL_JOBLIB)

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextIn):
    try:
        res = MODEL_PIPELINE(inp.text)

        if isinstance(res, list) and len(res) > 0:
            hf_label = res[0].get("label")   # LABEL_0 / LABEL_1 / LABEL_2
            score = float(res[0].get("score"))
            mapped_label = LABEL_MAP.get(hf_label, hf_label)  # fallback kalau tidak ditemukan

            return {
                "label": mapped_label,
                "hf_label": hf_label,   # opsional, bisa dihapus
                "score": score
            }
        else:
            raise HTTPException(status_code=500, detail="Model returned unexpected format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    import pandas as pd
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'text' column")

    preds = [MODEL_PIPELINE(t)[0] for t in df["text"].tolist()]

    df["hf_label"] = [p["label"] for p in preds]
    df["pred_label"] = [LABEL_MAP.get(p["label"], p["label"]) for p in preds]
    df["pred_score"] = [p["score"] for p in preds]

    out_path = Path("data/predictions")
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "predictions_batch.csv"
    df.to_csv(save_path, index=False)
    return FileResponse(save_path)

@app.get("/metrics")
def get_metrics():
    metrics_path = Path(EVAL_DIR) / "eval_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Eval metrics not found. Jalankan evaluate dulu.")
    return JSONResponse(content=json.load(open(metrics_path)))

@app.get("/classification_report")
def get_class_report():
    p = Path(EVAL_DIR) / "classification_report.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="classification_report.csv not found.")
    return FileResponse(p)

@app.get("/confusion_matrix")
def get_confusion_image():
    p = Path(EVAL_DIR) / "confusion_matrix.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="confusion_matrix.png not found.")
    return FileResponse(p)

@app.get("/download_model")
def download_model():
    from shutil import make_archive
    zip_path = Path("results_indobertweet") / "best_model.zip"
    if not zip_path.exists():
        make_archive(str(zip_path.with_suffix("")), 'zip', HF_MODEL_DIR)
    return FileResponse(zip_path, media_type="application/zip", filename="best_model.zip")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

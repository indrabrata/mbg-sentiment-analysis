from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import mlflow
import mlflow.pyfunc
import mlflow.transformers
from dotenv import load_dotenv

load_dotenv()

# Configure AWS/S3 credentials for MLflow artifact storage
# These MUST be set if MLflow uses remote artifact storage (S3/MinIO)
if os.getenv("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
if os.getenv("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mbg-sentiment-analysis")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", None)
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

PREDICTIONS_OUTPUT_DIR = os.getenv("PREDICTIONS_OUTPUT_DIR", "data/predictions")

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

app = FastAPI(title="Sentiment API (IndoBERTweet)")

MODEL_METADATA = {
    "source": "unknown",
    "run_id": None,
    "model_name": None,
    "loaded_at": None
}

@app.on_event("startup")
def load_model():
    global MODEL_PIPELINE
    from datetime import datetime
    MODEL_METADATA["loaded_at"] = datetime.now().isoformat()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")

    if MODEL_NAME:
        try:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            print(f"Loading model from registry: {model_uri}")
            MODEL_PIPELINE = mlflow.pyfunc.load_model(model_uri)
            MODEL_METADATA["source"] = "mlflow_registry"
            MODEL_METADATA["model_name"] = MODEL_NAME
            MODEL_METADATA["stage"] = MODEL_STAGE
            print(f"✅ Model loaded from MLflow registry: {MODEL_NAME}/{MODEL_STAGE}")
            return
        except Exception as e:
            error_msg = f"Failed to load from registry: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if not experiment:
            raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found in MLflow")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise RuntimeError(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'")

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading model from latest run: {run_id}")

        # Download artifacts once to avoid multiple downloads
        import tempfile
        from transformers import pipeline as hf_pipeline

        print("Downloading model artifacts...")
        temp_dir = tempfile.mkdtemp()
        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=temp_dir
        )
        print(f"✅ Artifacts downloaded to: {model_path}")

        # Now try loading with the downloaded artifacts
        # Load the model directly using transformers pipeline
        MODEL_PIPELINE = hf_pipeline("text-classification", model=model_path)
        print(f"✅ Model loaded successfully")

        MODEL_METADATA["source"] = "mlflow_run"
        MODEL_METADATA["run_id"] = run_id
        MODEL_METADATA["experiment_name"] = MLFLOW_EXPERIMENT_NAME
        print(f"✅ Model loaded from MLflow run: {run_id}")

    except Exception as e:
        error_msg = f"Failed to load model from MLflow: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

class TextIn(BaseModel):
    text: str

def _predict_single(text: str):
    """Helper function to handle both MLflow and local model inference"""
    try:
        if hasattr(MODEL_PIPELINE, 'predict'):
            import pandas as pd
            result = MODEL_PIPELINE.predict(pd.DataFrame({'text': [text]}))
            if isinstance(result, list):
                return result[0]
            return result
        else:
            return MODEL_PIPELINE(text)
    except Exception as e:
        return MODEL_PIPELINE(text)

@app.post("/predict")
def predict(inp: TextIn):
    try:
        res = _predict_single(inp.text)

        if isinstance(res, list) and len(res) > 0:
            hf_label = res[0].get("label")  
            score = float(res[0].get("score"))
            mapped_label = LABEL_MAP.get(hf_label, hf_label)  

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

    preds = []
    for text in df["text"].tolist():
        result = _predict_single(text)
        if isinstance(result, list) and len(result) > 0:
            preds.append(result[0])
        else:
            preds.append(result)

    df["hf_label"] = [p["label"] for p in preds]
    df["pred_label"] = [LABEL_MAP.get(p["label"], p["label"]) for p in preds]
    df["pred_score"] = [p["score"] for p in preds]

    out_path = Path("data/predictions")
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "predictions_batch.csv"
    df.to_csv(save_path, index=False)
    return FileResponse(save_path)

@app.get("/model_info")
def get_model_info():
    """Get information about the currently loaded model"""
    return {
        "model_metadata": MODEL_METADATA,
        "mlflow_config": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

# Todo : 
# 1. Create connection to postgres and saved data prediction with needed field
# 2. Create logging that will showed into the grafana

# Scripts Usage Guide

This guide explains how to use each script in the `scripts/` directory.

## Prerequisites

Make sure you're in the `model/` directory before running any scripts:
```bash
cd model
```

---

## 1. Data Preprocessing (`preprocess_data.py`)

**Purpose**: Clean and preprocess raw Twitter data (scraped text)

**Usage**:
```bash
python scripts/preprocess_data.py --input <input_csv> --output_dir <output_directory>
```

**Parameters**:
- `--input` (required): Path to raw CSV file with Twitter data
- `--output_dir` (required): Directory to save cleaned data

**Example**:
```bash
python scripts/preprocess_data.py \
  --input ./data/raw/mbg_scraped_2024-12-16.csv \
  --output_dir ./data/cleaned/
```

**Output**: Creates `mbg_cleaned.csv` in the output directory with:
- Lowercased text
- URLs replaced with `HTTPURL`
- Mentions replaced with `@USER`
- Emojis converted to text
- Cleaned whitespace

---

## 2. Add Label Column (`add_label_column.py`)

**Purpose**: Add empty label column to cleaned data for expert annotation

**Usage**:
```bash
python scripts/add_label_column.py --input <cleaned_csv> --output_dir <output_directory> [--date <date>]
```

**Parameters**:
- `--input` (required): Path to cleaned CSV file
- `--output_dir` (required): Directory to save output
- `--date` (optional): Date string (YYYY-MM-DD), defaults to today

**Example**:
```bash
python scripts/add_label_column.py \
  --input ./data/cleaned/mbg_cleaned.csv \
  --output_dir ./data/unlabeled/ \
  --date 2024-12-16
```

**Output**: Creates `mbg_cleaned_unlabeled_YYYY-MM-DD.csv` with empty `label` column

---

## 3. Merge Weekly Data (`merge_weekly_data.py`)

**Purpose**: Merge daily labeled files from the past 7 days into a weekly dataset

**Usage**:
```bash
python scripts/merge_weekly_data.py [--input_dir <input_directory>] [--output_dir <output_directory>]
```

**Parameters**:
- `--input_dir` (optional): Directory with daily labeled files (default: `./data/labeled/daily`)
- `--output_dir` (optional): Directory to save weekly merged file (default: `./data/labeled/weekly`)

**Example**:
```bash
python scripts/merge_weekly_data.py \
  --input_dir ./data/labeled/daily \
  --output_dir ./data/labeled/weekly
```

**Expected Input Files**:
- Files named like `mbg_labeled_2024-12-16.csv` in the input directory

**Output**: Creates `mbg_labelled_week_<week_number>.csv`

---

## 4. Train Model (`train.py`)

**Purpose**: Incremental training of IndoBERTweet model with layer freezing

**Usage**:
```bash
python scripts/train.py \
  --train <train_csv> \
  --val <validation_csv> \
  --model_dir <previous_model_directory> \
  --output_dir <output_directory> \
  [--epochs <num_epochs>] \
  [--freeze_layers <num_layers>]
```

**Parameters**:
- `--train` (required): Path to training CSV (weekly expert-labeled data)
- `--val` (required): Path to validation CSV
- `--model_dir` (required): Path to previous model directory to continue training from
- `--output_dir` (required): Directory to save trained model
- `--epochs` (optional): Number of training epochs (default: 2)
- `--freeze_layers` (optional): Number of bottom encoder layers to freeze (default: 6)

**Example**:
```bash
# Set MLflow environment variables first
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-password

python scripts/train.py \
  --train ./data/labeled/weekly/mbg_labelled_week_50.csv \
  --val ./data/processed/val.csv \
  --model_dir ./models/previous \
  --output_dir ./models/week_50 \
  --epochs 3 \
  --freeze_layers 6
```

**Required Environment Variables**:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MLFLOW_TRACKING_USERNAME`: MLflow username
- `MLFLOW_TRACKING_PASSWORD`: MLflow password

**Output**:
- Trained model saved to output directory
- Metrics logged to MLflow

---

## 5. Evaluate Model (`evaluate.py`)

**Purpose**: Evaluate trained model and generate metrics/visualizations

**Usage**:
```bash
python scripts/evaluate.py \
  --test_path <test_csv> \
  --model_dir <model_directory> \
  --output_dir <output_directory>
```

**Parameters**:
- `--test_path` (required): Path to test CSV file
- `--model_dir` (required): Path to trained model directory
- `--output_dir` (required): Directory to save evaluation results

**Example**:
```bash
python scripts/evaluate.py \
  --test_path ./data/processed/test.csv \
  --model_dir ./models/week_50 \
  --output_dir ./results/week_50_eval
```

**Output Files**:
- `eval_metrics.json`: Accuracy, precision, recall, F1 score
- `classification_report.csv`: Per-class metrics
- `confusion_matrix.csv`: Confusion matrix data
- `confusion_matrix.png`: Confusion matrix visualization

---

## 6. Model Registry (`model_registry.py`)

**Purpose**: Download or push models to/from MLflow registry

### Download Latest Model

**Usage**:
```bash
python scripts/model_registry.py download \
  [--model_name <model_name>] \
  [--target_dir <target_directory>]
```

**Parameters**:
- `--model_name` (optional): Model name in registry (default: "MBGModel")
- `--target_dir` (optional): Directory to save downloaded model (default: "models/previous")

**Example**:
```bash
# Set MLflow environment variables first
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-password

python scripts/model_registry.py download \
  --model_name MBGModel \
  --target_dir ./models/previous
```

### Push Model to Registry

**Usage**:
```bash
python scripts/model_registry.py push \
  --model_dir <model_directory> \
  [--model_name <model_name>] \
  [--stage <stage>]
```

**Parameters**:
- `--model_dir` (required): Path to model directory to push
- `--model_name` (optional): Model name in registry (default: "MBGModel")
- `--stage` (optional): Stage to set (default: "Staging", options: "Staging", "Production")

**Example**:
```bash
python scripts/model_registry.py push \
  --model_dir ./models/week_50 \
  --model_name MBGModel \
  --stage Staging
```

---

## Complete Workflow Example

Here's a typical end-to-end workflow:

```bash
# 1. Preprocess raw data
python scripts/preprocess_data.py \
  --input ./data/raw/mbg_scraped_2024-12-16.csv \
  --output_dir ./data/cleaned/

# 2. Add label column for expert annotation
python scripts/add_label_column.py \
  --input ./data/cleaned/mbg_cleaned.csv \
  --output_dir ./data/unlabeled/ \
  --date 2024-12-16

# 3. Expert labels the data manually (outside this workflow)
#    Save as: ./data/labeled/daily/mbg_labeled_2024-12-16.csv

# 4. Merge weekly data (after accumulating daily files)
python scripts/merge_weekly_data.py \
  --input_dir ./data/labeled/daily \
  --output_dir ./data/labeled/weekly

# 5. Download previous model from registry
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-password

python scripts/model_registry.py download \
  --model_name MBGModel \
  --target_dir ./models/previous

# 6. Train model incrementally
python scripts/train.py \
  --train ./data/labeled/weekly/mbg_labelled_week_50.csv \
  --val ./data/processed/val.csv \
  --model_dir ./models/previous \
  --output_dir ./models/week_50 \
  --epochs 3 \
  --freeze_layers 6

# 7. Evaluate model
python scripts/evaluate.py \
  --test_path ./data/processed/test.csv \
  --model_dir ./models/week_50 \
  --output_dir ./results/week_50_eval

# 8. Push model to staging
python scripts/model_registry.py push \
  --model_dir ./models/week_50 \
  --model_name MBGModel \
  --stage Staging

# 9. After validation, promote to production (optional)
python scripts/model_registry.py push \
  --model_dir ./models/week_50 \
  --model_name MBGModel \
  --stage Production
```

---

## Notes

1. **Working Directory**: Always run scripts from the `model/` directory
2. **MLflow**: Training and model registry scripts require MLflow environment variables
3. **Data Format**: CSV files must have appropriate columns:
   - Raw data: `full_text`, `lang` (optional)
   - Training data: `text`, `label` (0=negative, 1=neutral, 2=positive)
4. **File Naming**: Some scripts expect specific file naming patterns (e.g., `mbg_labeled_YYYY-MM-DD.csv`)

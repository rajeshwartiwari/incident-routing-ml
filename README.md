# Incident Routing ML
Machine learning pipeline and FastAPI microservice to automatically classify and route incoming IT incidents to the right resolver team.

## Table of Contents
- [Project Overview & Business Objectives](#project-overview--business-objectives)
- [Success Metrics](#success-metrics)
- [Incident Routing ML Lifecycle](#incident-routing-ml-lifecycle)
  - [1. Data Collection & Leakage Prevention](#1-data-collection--leakage-prevention)
  - [2. Data Validation & Preparation](#2-data-validation--preparation)
  - [3. Feature Engineering & Feature Store](#3-feature-engineering--feature-store)
  - [4. Time-Based Dataset Splitting](#4-time-based-dataset-splitting)
  - [5. Baseline Models & Training](#5-baseline-models--training)
  - [6. Experiment Tracking with MLflow](#6-experiment-tracking-with-mlflow)
  - [7. Evaluation Metrics & Error Analysis](#7-evaluation-metrics--error-analysis)
  - [8. Model Governance & Registry](#8-model-governance--registry)
  - [9. Shadow Mode Deployment](#9-shadow-mode-deployment)
  - [10. Monitoring, Drift & Continuous Improvement](#10-monitoring-drift--continuous-improvement)
  - [11. Progressive Rollout: Assisted to Automated Routing](#11-progressive-rollout-assisted-to-automated-routing)
- [Quickstart & Usage](#quickstart--usage)
  - [Generate Synthetic Data](#generate-synthetic-data)
  - [Train Naive Model](#train-naive-model)
  - [Train & Track with MLflow](#train--track-with-mlflow)
  - [Run Evaluation (Top-1 & Top-3)](#run-evaluation-top-1--top-3)
  - [Run FastAPI Service](#run-fastapi-service)
  - [Shadow Mode & Monitoring](#shadow-mode--monitoring)
- [Repository Structure](#repository-structure)

---

## Project Overview & Business Objectives

When IT service desk incidents are misrouted, resolution times soar, mean time to resolve (MTTR) increases, and analyst toil multiplies through repeated ticket ping-pong. 

The primary business objectives of this ML platform are:
- **Reduce Incident Reassignments:** Minimize unnecessary ticket handoffs across resolver groups (e.g. Service Desk -> End User Computing -> Application Support).
- **Improve First-Time-Right (FTR) Routing:** Route tickets to the correct resolver group on the initial triage step.
- **Accelerate Resolution Times:** Reduce MTTR by eliminating the initial manual dispatch bottleneck.

---

## Success Metrics

We measure model and operational success across offline and online dimensions:

| Metric | Target | Description |
|---|---|---|
| **Top-1 Routing Accuracy** | $\ge 85\%$ | Model's top predicted assignment group matches the actual resolving group. |
| **Top-3 Accuracy** | $\ge 95\%$ | True resolving group is within the model's top 3 ranked recommendations (crucial for analyst-assisted mode). |
| **Macro F1-Score** | $\ge 0.80$ | Unweighted average F1 across all assignment groups, preventing neglect of smaller specialist teams. |
| **Weighted F1-Score** | $\ge 0.85$ | Class-size weighted F1 across all incident categories. |
| **Reassignment Rate Reduction** | $\ge 25\%$ decrease | Operational metric tracking tickets transferred $\ge 2$ times post-triage. |
| **First-Contact Resolution (FCR) Rate** | Operational uplift | Incidents handled promptly by the initial assignment team. |

---

## Incident Routing ML Lifecycle

### 1. Data Collection & Leakage Prevention
- **Historical Extraction:** Extract historical incident records from IT Service Management platforms (e.g., ServiceNow `incident` table), including incident ID, timestamps, description, urgency, impact, priority, category, and final resolving group (`assignment_group`).
- **Leakage Prevention Rule:** Only include features available **at ticket creation time**. Fields populated during investigation or resolution (such as resolution notes, close codes, re-assignment logs, duration, root-cause category assigned retroactively) **must be excluded** to prevent severe target leakage.

### 2. Data Validation & Preparation
- **Deduplication & Completeness:** Drop duplicate incidents, empty descriptions, or records with missing routing targets.
- **Label Consistency:** Filter out deprecated or merged assignment groups, ensuring consistency with the current IT organizational hierarchy.
- **Rule-based & Great Expectations Checks:** Enforce constraints on description text length, valid urgency levels (`Low`, `Medium`, `High`), and impact levels.

### 3. Feature Engineering & Feature Store
- **Text Cleaning & Standardisation:** Normalise free-text descriptions (lower-casing, URL/ticket ID/email scrubbing, whitespace normalisation).
- **Text Representation:** Extract unigram and bigram TF-IDF representations (or dense sentence embeddings) capturing domain terms and IT failure modes.
- **Categorical Encoding:** One-hot encode priority, urgency, and impact.
- **Feature Store Integration:** Store reproducible offline/online feature definitions in Feast (`feature_repo`) to eliminate training-serving skew.

### 4. Time-Based Dataset Splitting
- **Chronological Split:** Random splits cause temporal leakage and overly optimistic performance estimates. Incidents are sorted by `created_at`.
- **Train / Validation / Test:** Earlier historical incidents form the training set (e.g., 80%), intermediate periods form validation, and the most recent incident period is reserved as the unseen test set to simulate true production conditions.

### 5. Baseline Models & Training
- **Benchmark Baselines:** Start with linear benchmarks—**Logistic Regression** and **Linear Support Vector Machines (Linear SVM)** with TF-IDF features—before introducing complex deep learning models.
- **Pipeline Architecture:** Scikit-learn `Pipeline` encapsulating `ColumnTransformer` (TF-IDF vectorizer + OneHotEncoder) with the classifier ensures all transformations fit strictly on training splits.

### 6. Experiment Tracking with MLflow
- **Artifact & Metric Logging:** Log parameters ($C$, n-gram ranges, solver, kernel), training/test metrics, feature lists, and dataset sizes to an MLflow Tracking Server (`http://localhost:5001`).
- **Model Signatures & Run Tags:** Log trained pipelines with MLflow model signatures and input examples, saving confusion matrices as versioned artifacts.

### 7. Evaluation Metrics & Error Analysis
- **Comprehensive Multi-class Evaluation:** Evaluate Top-1 Accuracy, Top-3 Accuracy, Precision, Recall, Macro F1, Weighted F1, and multi-class One-vs-Rest ROC-AUC.
- **Per-Group Confusion Matrix & Error Slices:** Inspect misclassification patterns between overlapping queues (e.g., *Service Desk* vs. *Application Support* or *Network* vs. *Database Operations*).

### 8. Model Governance & Registry
- **Model Registry Promotion:** Approved candidate models are registered in the MLflow Model Registry (`incident-routing-model`).
- **Sign-Off Workflow:** Formal technical validation, security review (no PII or credentials retained), and business stakeholder sign-off prior to assigning the `@champion` or production alias.

### 9. Shadow Mode Deployment
- **Silent Inference in Production:** Run alongside the live triage process without automatically routing tickets.
- **Silent Audit:** Compare model recommendations in real time against human triage decisions to build operational confidence without risking operational disruption.

### 10. Monitoring, Drift & Continuous Improvement
- **Performance & Drift Tracking:** Monitor real-time prediction confidence distributions, reassignment rates, and input text drift using statistical tests (Kolmogorov-Smirnov / Evidently).
- **Retraining Triggers:** Automated retraining alerts trigger when confidence decays or feature drift exceeds designated statistical thresholds.

### 11. Progressive Rollout: Assisted to Automated Routing
- **Phase 1 — Shadow Mode:** Evaluate real-world agreement rate silently.
- **Phase 2 — Analyst-Assisted Recommendations:** Present the model's Top-3 recommendations with confidence scores inside ServiceNow to assist service desk triage agents.
- **Phase 3 — Automated Routing for High-Confidence Tickets:** Directly route incidents whose predicted confidence exceeds a strict threshold (e.g., $\ge 90\%$), falling back to human triage otherwise.

---

## Quickstart & Usage

### Generate Synthetic Data
```bash
# Time-sorted split (80% train, 20% test)
python -m src.generate_synthetic_data --rows 1000 --seed 42 --output-dir data/raw
```

### Train Naive Model
Trains a baseline Logistic Regression model and saves `models/saved_models/model.pkl`:
```bash
python -m src.train_naive
```

### Train & Track with MLflow
Runs a baseline hyperparameter sweep with Logistic Regression and Linear SVM models tracked in MLflow:
```bash
# Start MLflow server (optional local UI)
mlflow server --host 0.0.0.0 --port 5001

# Run experiments & register best model to Model Registry
python -m src.train_mlflow
```

### Run Evaluation (Top-1 & Top-3)
Evaluate models and generate detailed classification reports, confusion matrices, and Top-3 accuracy:
```bash
python -m src.evaluate
```

### Run FastAPI Service
```bash
# 1. Naive server (local pickle):
python -m uvicorn src.serve_naive:app --reload --port 8001

# 2. MLflow Champion server (loads @champion from MLflow registry):
python -m uvicorn src.serve_mlflow:app --reload --port 8002
```

Example request:
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "User account locked out in Active Directory after password expiry",
    "urgency": "High",
    "impact": "Medium"
  }'
```

---

## Repository Structure

```
├── data/
│   └── raw/                   # train.csv, test.csv
├── feature_repo/              # Feast feature store definitions
├── models/
│   └── saved_models/          # Exported model pipelines (.pkl)
├── src/
│   ├── generate_synthetic_data.py # Synthetic generator with time-based splits
│   ├── preprocess.py              # Text cleaning & validation helpers
│   ├── train_naive.py             # Baseline local training script
│   ├── train_mlflow.py            # MLflow tracking with Logistic Regression & Linear SVM
│   ├── evaluate.py                # Top-1, Top-3, and per-group error analysis
│   ├── serve_naive.py             # FastAPI prediction service (Local artifact)
│   ├── serve_mlflow.py            # FastAPI service loading @champion from MLflow
│   ├── test_bad_data.py           # Robustness test cases & input fuzzing
│   └── monitoring.py              # Drift detection and prediction confidence monitoring
└── tests/                         # Unit tests
```


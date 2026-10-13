# src/serve_mlflow.py
"""
Serve the incident routing model from MLflow Model Registry.

This version loads the @champion model from MLflow, which means:
- Always serves the latest @champion model
- Can roll back by changing the @champion alias
- No manual file copying needed
"""
from typing import Dict, List, Literal, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess import validate_incident_payload

MODEL_NAME = "incident-routing-model"
TRACKING_URI = "http://localhost:5001"
AUTO_ROUTING_THRESHOLD = 0.85

# Configure MLflow
mlflow.set_tracking_uri(TRACKING_URI)

print("Loading model from MLflow Model Registry...")

# Load the champion model from the registry
# This automatically gets whichever version has the @champion alias
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    print("Successfully loaded champion model from MLflow!")
except Exception as e:
    print(f"Error loading from MLflow: {e}")
    print("Make sure you've assigned the @champion alias to a model in the MLflow UI")
    raise

# No separate encoder: the registered pipeline includes TF-IDF and one-hot encoding.

app = FastAPI(
    title="Incident Routing API (MLflow)",
    description="""
    Incident routing API that loads models from MLflow Model Registry.
    
    Supports:
    - Always serves the champion model (@champion alias)
    - Top-1 and Top-3 recommendations for analyst-assisted routing
    - Shadow mode deployment evaluation against live analyst decisions
    - Progressive automated routing thresholding
    """,
    version="2.1.0"
)

Level = Literal["Low", "Medium", "High"]


class Incident(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the incident",
        example="User cannot access mailbox after password reset",
    )
    urgency: Level = Field(..., description="Urgency of the incident", example="High")
    impact: Level = Field(..., description="Impact of the incident", example="Medium")
    actual_analyst_group: Optional[str] = Field(
        None,
        description="Actual assignment group chosen by analyst (for shadow-mode auditing)",
    )


class Recommendation(BaseModel):
    assignment_group: str
    confidence: float


class PredictionResponse(BaseModel):
    assignment_group: str
    confidence: float
    top_3_recommendations: List[Recommendation]
    routing_mode: str
    shadow_mode: bool
    shadow_agreement: Optional[bool] = None
    probabilities: Dict[str, float]
    model_source: str = "MLflow Production"


@app.post("/predict", response_model=PredictionResponse)
def predict(incident: Incident, shadow_mode: bool = False):
    """Predict the assignment group for an incident using the champion model."""
    is_valid, validation_errors = validate_incident_payload(incident.dict())
    if not is_valid:
        raise HTTPException(status_code=422, detail={"errors": validation_errors})

    # Same column names and order as used in training
    X = pd.DataFrame(
        [[incident.description, incident.urgency, incident.impact]],
        columns=["description", "urgency", "impact"],
    )

    probabilities = model.predict_proba(X)[0]
    classes = model.classes_
    best = int(probabilities.argmax())
    top_confidence = round(float(probabilities[best]), 4)
    top_group = str(classes[best])

    k_val = min(3, len(classes))
    top_indices = np.argsort(probabilities)[-k_val:][::-1]
    top_3 = [
        Recommendation(
            assignment_group=str(classes[idx]),
            confidence=round(float(probabilities[idx]), 4),
        )
        for idx in top_indices
    ]

    routing_mode = "automated" if top_confidence >= AUTO_ROUTING_THRESHOLD else "analyst_assisted"
    agreement = None
    if shadow_mode and incident.actual_analyst_group:
        agreement = top_group.lower().strip() == incident.actual_analyst_group.lower().strip()

    return PredictionResponse(
        assignment_group=top_group,
        confidence=top_confidence,
        top_3_recommendations=top_3,
        routing_mode=routing_mode,
        shadow_mode=shadow_mode,
        shadow_agreement=agreement,
        probabilities={
            str(group): round(float(p), 4)
            for group, p in zip(classes, probabilities)
        },
        model_source="MLflow Production",
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_source": "MLflow Registry"}

@app.get("/model-info")
def model_info():
    """Get information about the currently loaded model."""
    return {
        "registry": "MLflow",
        "model_name": MODEL_NAME,
        "alias": "champion",
        "tracking_uri": TRACKING_URI
    }
# src/serve_naive.py
"""
Serve the incident routing model as a REST API - NAIVE VERSION.

This is a simple API that:
1. Loads the trained model at startup
2. Accepts incident details via POST request
3. Returns the predicted assignment group

No monitoring or model versioning yet.
"""
import pickle
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess import validate_incident_payload

# Loaded once at startup. The pickle is a full pipeline (TF-IDF + one-hot
# encoding + classifier), so no separate encoder is needed.
print("Loading model...")
with open("models/saved_models/model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

app = FastAPI(
    title="Incident Routing API",
    description="""
    Predict which assignment group should handle an IT incident.

    Supports:
    - Top-1 assignment group recommendation with confidence score
    - Top-3 recommended groups for analyst-assisted routing
    - Automated routing threshold check (e.g. >= 0.85 confidence)
    - Shadow mode deployment toggle
    """,
    version="1.1.0",
)

Level = Literal["Low", "Medium", "High"]


class Incident(BaseModel):
    """Schema for an incident to be routed."""
    description: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the incident",
        example="User cannot access mailbox after password reset",
    )
    urgency: Level = Field(..., description="Urgency of the incident", example="High")
    impact: Level = Field(..., description="Impact of the incident", example="Medium")
    # Optional operational metadata passed by caller
    actual_analyst_group: Optional[str] = Field(
        None,
        description="Actual assignment group assigned by human analyst (used for shadow-mode auditing)",
    )


class Recommendation(BaseModel):
    assignment_group: str
    confidence: float


class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    assignment_group: str = Field(description="Top-1 predicted assignment group")
    confidence: float = Field(description="Probability of the top predicted group (0.0 to 1.0)")
    top_3_recommendations: List[Recommendation] = Field(
        description="Top-3 ranked assignment groups for analyst-assisted routing"
    )
    routing_mode: str = Field(
        description="Recommended action: 'automated' (if confidence >= threshold), else 'analyst_assisted'"
    )
    shadow_mode: bool = Field(
        default=False,
        description="If True, recommendation was audited silently without auto-updating ServiceNow",
    )
    shadow_agreement: Optional[bool] = Field(
        None,
        description="If actual_analyst_group provided in shadow mode, indicates if model agreed",
    )
    probabilities: Dict[str, float] = Field(description="Probability for each assignment group")


AUTO_ROUTING_THRESHOLD = 0.85


@app.post("/predict", response_model=PredictionResponse)
def predict(incident: Incident, shadow_mode: bool = False):
    """Predict the assignment group for an incident."""
    # Data validation & leakage screening
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

    # Compute Top-3 recommendations
    k_val = min(3, len(classes))
    top_indices = np.argsort(probabilities)[-k_val:][::-1]
    top_3 = [
        Recommendation(
            assignment_group=str(classes[idx]),
            confidence=round(float(probabilities[idx]), 4),
        )
        for idx in top_indices
    ]

    # Progressive automation logic
    routing_mode = "automated" if top_confidence >= AUTO_ROUTING_THRESHOLD else "analyst_assisted"

    # Shadow-mode audit comparison
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
    )


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Incident Routing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

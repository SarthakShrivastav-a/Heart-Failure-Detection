from pydantic import BaseModel, Field
from typing import Optional


class HeartFailurePredictionRequest(BaseModel):
    """Request model for heart failure prediction"""
    age: float = Field(..., description="Age of the patient", ge=0, le=120)
    anemia: int = Field(..., description="Presence of anemia (0 or 1)", ge=0, le=1)
    creatinine_phosphokinase: float = Field(..., description="Level of CPK enzyme in mcg/L", ge=0)
    diabetes: int = Field(..., description="Presence of diabetes (0 or 1)", ge=0, le=1)
    ejection_fraction: float = Field(..., description="Percentage of blood leaving heart at each contraction", ge=0, le=100)
    high_blood_pressure: int = Field(..., description="Presence of high blood pressure (0 or 1)", ge=0, le=1)
    platelets: float = Field(..., description="Platelets in blood (kiloplatelets/mL)", ge=0)
    serum_creatinine: float = Field(..., description="Level of serum creatinine in mg/dL", ge=0)
    serum_sodium: float = Field(..., description="Level of serum sodium in mEq/L", ge=0)
    sex: int = Field(..., description="Sex of the patient (0 = female, 1 = male)", ge=0, le=1)
    smoking: int = Field(..., description="Smoking status (0 or 1)", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 75.0,
                "anemia": 0,
                "creatinine_phosphokinase": 582,
                "diabetes": 0,
                "ejection_fraction": 20,
                "high_blood_pressure": 1,
                "platelets": 265000.00,
                "serum_creatinine": 1.9,
                "serum_sodium": 130,
                "sex": 1,
                "smoking": 0
            }
        }


class HeartFailurePredictionResponse(BaseModel):
    """Response model for heart failure prediction"""
    prediction: int = Field(..., description="Predicted death event (0 = No, 1 = Yes)")
    prediction_probability: float = Field(..., description="Probability of death event (0-1)")
    message: str = Field(..., description="Human-readable prediction message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_probability": 0.85,
                "message": "High risk of death event detected"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Health check message")


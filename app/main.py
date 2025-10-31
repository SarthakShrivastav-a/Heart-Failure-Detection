from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.models import (
    HeartFailurePredictionRequest,
    HeartFailurePredictionResponse,
    HealthResponse
)
from app.prediction import get_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Failure Prediction API",
    description="API for predicting heart failure death events using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    try:
        logger.info("Initializing predictor...")
        get_predictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="ok",
        message="Heart Failure Prediction API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            message="Service is operational and ready for predictions"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )


@app.post("/predict", response_model=HeartFailurePredictionResponse)
async def predict_heart_failure(request: HeartFailurePredictionRequest):
    """
    Predict heart failure death event based on patient features
    
    Args:
        request: HeartFailurePredictionRequest containing patient data
        
    Returns:
        HeartFailurePredictionResponse with prediction and probability
    """
    try:
        # Convert request to dictionary
        features = request.model_dump()
        
        # Get predictor and make prediction
        predictor = get_predictor()
        prediction, probability, message = predictor.predict(features)
        
        return HeartFailurePredictionResponse(
            prediction=prediction,
            prediction_probability=probability,
            message=message
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


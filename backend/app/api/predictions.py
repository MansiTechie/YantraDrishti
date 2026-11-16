import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.prediction import (
    RULPredictionRequest,
    RULPredictionResponse,
)
from app.services.inference_service import (
    InvalidPredictionError,
    ModelUnavailableError,
    RULInferenceService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Predictions"],
)

# Shared unconfigured default service instance (lazy loading, no startup crash)
_default_inference_service = RULInferenceService()


def get_inference_service() -> RULInferenceService:
    """Dependency provider returning the active RULInferenceService instance."""
    return _default_inference_service


@router.post(
    "/rul",
    response_model=RULPredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "RUL prediction successfully computed."},
        422: {"description": "Validation Error (invalid or missing request features)."},
        500: {"description": "Internal error during prediction computation."},
        503: {"description": "Trained RUL model is currently unavailable."},
    },
)
def predict_rul(
    request: RULPredictionRequest,
    service: RULInferenceService = Depends(get_inference_service),
) -> RULPredictionResponse:
    """
    Predict Remaining Useful Life (RUL) in cycles for a machine given its current
    telemetry and cycle-to-cycle sensor deltas (46 canonical features).
    """
    try:
        return service.predict(request)
    except ModelUnavailableError as e:
        logger.warning("RUL prediction requested but model is unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RUL prediction model is currently unavailable.",
        )
    except InvalidPredictionError as e:
        logger.error("Invalid prediction computation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while calculating the RUL prediction.",
        )
    except Exception as e:
        logger.error("Unexpected prediction failure: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during prediction.",
        )

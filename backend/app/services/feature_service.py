import math

from app.schemas.prediction import (
    FEATURE_ORDER,
    RULPredictionRequest,
)

FEATURE_COUNT: int = len(FEATURE_ORDER)  # 46


def extract_feature_dict(request: RULPredictionRequest) -> dict[str, float]:
    """
    Extracts an ordered dictionary of 46 model features from an RULPredictionRequest.

    The order strictly matches FEATURE_ORDER:
      1. cycle (1)
      2. setting_1, setting_2, setting_3 (3)
      3. sensor_01 through sensor_21 (21)
      4. sensor_01_delta through sensor_21_delta (21)

    machine_id is strictly excluded.
    """
    feature_dict = request.to_feature_dict()

    if len(feature_dict) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, got {len(feature_dict)}"
        )

    for name, value in feature_dict.items():
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite value encountered in feature '{name}': {value}"
            )

    return feature_dict


def extract_feature_vector(request: RULPredictionRequest) -> list[float]:
    """
    Extracts an ordered list of 46 numerical float features from an RULPredictionRequest
    suitable for direct input to an ML inference pipeline.

    Order strictly matches FEATURE_ORDER.
    machine_id is strictly excluded.
    """
    feature_dict = extract_feature_dict(request)
    return [feature_dict[feature_name] for feature_name in FEATURE_ORDER]

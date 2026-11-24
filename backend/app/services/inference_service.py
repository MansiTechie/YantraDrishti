import math
import pickle
from pathlib import Path
from typing import Any, Optional

from app.schemas.prediction import (
    FEATURE_ORDER,
    PredictionStatus,
    RULPredictionRequest,
    RULPredictionResponse,
)
from app.services.feature_service import extract_feature_vector


class ModelUnavailableError(Exception):
    """Raised when model artifact is missing, unconfigured, or cannot be loaded."""
    pass


class InvalidPredictionError(Exception):
    """Raised when a model produces a negative, non-finite, or malformed prediction."""
    pass


class RULInferenceService:
    """
    RUL Inference Service.

    Executes model inference on a validated RULPredictionRequest using
    the canonical 46-feature representation. Does not load artifacts at import
    time, ensuring the application starts cleanly even if artifacts are pending.
    """

    def __init__(
        self,
        model_artifact_path: Optional[Path | str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        model_instance: Optional[Any] = None,
    ):
        self.artifact_path: Optional[Path] = (
            Path(model_artifact_path) if model_artifact_path is not None else None
        )
        self._model: Optional[Any] = model_instance
        self._model_name: Optional[str] = model_name
        self._model_version: Optional[str] = model_version

    @property
    def is_available(self) -> bool:
        """Returns True if a model is ready in memory or a valid artifact file exists on disk."""
        if self._model is not None and hasattr(self._model, "predict"):
            return bool(self._model_name and self._model_version)
        if self.artifact_path is not None and self.artifact_path.is_file():
            return True
        return False

    def load_model(self) -> None:
        """
        Loads the trained model artifact from disk.

        Raises ModelUnavailableError if path is unconfigured, file does not exist,
        loading fails, or the loaded artifact lacks required prediction capabilities.
        """
        if self._model is not None:
            return

        if self.artifact_path is None:
            raise ModelUnavailableError(
                "Model artifact path is not configured. Trained model artifact is currently unavailable."
            )

        if not self.artifact_path.is_file():
            raise ModelUnavailableError(
                f"Model artifact not found at configured path: '{self.artifact_path}'."
            )

        try:
            try:
                import joblib
                loaded = joblib.load(self.artifact_path)
            except Exception:
                with open(self.artifact_path, "rb") as f:
                    loaded = pickle.load(f)

            if isinstance(loaded, dict) and "model" in loaded:
                self._model = loaded["model"]
                self._model_name = self._model_name or loaded.get("model_name")
                self._model_version = self._model_version or loaded.get("model_version")

                # Validate feature-order compatibility if features are specified in the artifact
                saved_features = loaded.get("features")
                if saved_features is None or list(saved_features) != list(FEATURE_ORDER):
                    self._model = None
                    raise ModelUnavailableError(
                        f"Model artifact feature specification is missing or does not match canonical 46-feature ordering. "
                        f"Expected {len(FEATURE_ORDER)} features."
                    )
            else:
                self._model = loaded

            if not hasattr(self._model, "predict") or not callable(self._model.predict):
                self._model = None
                raise ModelUnavailableError(
                    f"Model artifact at '{self.artifact_path}' does not provide a callable 'predict' method."
                )

        except ModelUnavailableError:
            raise
        except Exception as e:
            self._model = None
            raise ModelUnavailableError(
                f"Failed to load model artifact from '{self.artifact_path}': {e}"
            ) from e

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse:
        """
        Runs RUL prediction for the given request.

        1. Ensures the model is loaded and ready.
        2. Validates that model metadata (name, version) is configured.
        3. Extracts the exact 46-feature vector in canonical order.
        4. Calls model.predict() with 2D input.
        5. Validates non-negative and finite prediction.
        6. Returns a validated RULPredictionResponse in units of cycles.
        """
        if self._model is None:
            self.load_model()

        if not self._model_name or not self._model_version:
            raise ModelUnavailableError(
                "Verified model metadata (model_name and model_version) is missing or unconfigured."
            )

        # Extract canonical 46-feature input vector
        features = extract_feature_vector(request)

        try:
            if hasattr(self._model, "feature_names_in_"):
                try:
                    import pandas as pd
                    model_input = pd.DataFrame([features], columns=self._model.feature_names_in_)
                except ImportError:
                    model_input = [features]
            else:
                model_input = [features]
            raw_output = self._model.predict(model_input)
        except Exception as e:
            raise InvalidPredictionError(f"Model prediction execution failed: {e}") from e

        # Extract numeric scalar
        if hasattr(raw_output, "__getitem__"):
            pred_val = float(raw_output[0])
        else:
            pred_val = float(raw_output)

        # Reject non-finite values
        if not math.isfinite(pred_val):
            raise InvalidPredictionError(
                f"Model returned non-finite RUL prediction: {pred_val}"
            )

        # Reject negative RUL values
        if pred_val < 0.0:
            raise InvalidPredictionError(
                f"Model returned negative RUL prediction: {pred_val} cycles"
            )

        return RULPredictionResponse(
            machine_id=request.machine_id,
            predicted_rul_cycles=pred_val,
            unit="cycles",
            model_name=self._model_name,
            model_version=self._model_version,
            prediction_status=PredictionStatus.SUCCESS,
        )


def get_default_model_path() -> Optional[Path]:
    """
    Resolves the default trained model artifact path if present on disk or specified via environment.
    Always returns a fully resolved absolute Path so it functions consistently regardless of CWD.
    """
    import os

    env_path = os.getenv("MODEL_ARTIFACT_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p

    # Absolute path based on repository root relative to this file:
    # backend/app/services/inference_service.py -> parents[3] is repository root
    repo_root = Path(__file__).resolve().parents[3]
    repo_artifact = (repo_root / "ml" / "models" / "rul_baseline.joblib").resolve()
    if repo_artifact.is_file():
        return repo_artifact

    # Fallback checking relative to current working directory
    cwd_artifact = Path("ml/models/rul_baseline.joblib").resolve()
    if cwd_artifact.is_file():
        return cwd_artifact

    return None

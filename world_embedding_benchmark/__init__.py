"""Lightweight tools for the World Embedding Benchmark."""

from .models import get_model, register_model
from .pair_classification import PairClassificationResult, evaluate_pair_classification
from .pair_classification_data import (
    build_pair_classification_manifest,
    create_pair_classification_manifest,
)
from .regression import (
    DEFAULT_ALPHAS,
    REGRESSION_CONFIGS,
    RegressionItem,
    RegressionResult,
    evaluate_video_regression,
    load_regression_data,
    nested_ridge_predictions,
    regression_metrics,
)
from .regression_scaling import (
    RegressionScalingResult,
    ScalingRunResult,
    evaluate_video_regression_scaling,
)
from .regression_splits import (
    DEFAULT_REPETITION_SEEDS,
    DEFAULT_TRAIN_SIZES,
    build_regression_split_manifest,
    create_regression_split_manifest,
    load_and_validate_regression_split_manifest,
)

from .retrieval import (
    DirectionalRetrievalResult,
    RetrievalResult,
    RetrievalSuiteResult,
    evaluate_text_video_retrieval,
    evaluate_text_video_retrieval_suite,
)

__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_REPETITION_SEEDS",
    "DEFAULT_TRAIN_SIZES",
    "REGRESSION_CONFIGS",
    "PairClassificationResult",
    "DirectionalRetrievalResult",
    "RegressionItem",
    "RegressionResult",
    "RegressionScalingResult",
    "RetrievalResult",
    "RetrievalSuiteResult",
    "ScalingRunResult",
    "build_pair_classification_manifest",
    "build_regression_split_manifest",
    "create_pair_classification_manifest",
    "create_regression_split_manifest",
    "evaluate_pair_classification",
    "evaluate_text_video_retrieval",
    "evaluate_text_video_retrieval_suite",
    "evaluate_video_regression",
    "evaluate_video_regression_scaling",
    "get_model",
    "load_regression_data",
    "load_and_validate_regression_split_manifest",
    "nested_ridge_predictions",
    "regression_metrics",
    "register_model",
]

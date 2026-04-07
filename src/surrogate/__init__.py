"""M4-M7 surrogate package — numpy-based MLP for American option pricing."""
from .data_gen import SurrogateTrainSample, generate_training_data, generate_training_data_stratified
from .mlp import AmericanPriceMLP
from .normalizer import FeatureNormalizer
from .train import TrainConfig, TrainHistory, train_surrogate
from .evaluate import (
    BucketEvalResult,
    FinancialConstraintMetrics,
    evaluate_financial_constraints,
    evaluate_surrogate,
    predict_price,
)
from .greeks import (
    SurrogateGreeks,
    load_surrogate_artifacts,
    surrogate_ad_greeks,
    surrogate_fd_greeks,
)

__all__ = [
    "AmericanPriceMLP",
    "BucketEvalResult",
    "FinancialConstraintMetrics",
    "SurrogateGreeks",
    "FeatureNormalizer",
    "SurrogateTrainSample",
    "TrainConfig",
    "TrainHistory",
    "evaluate_financial_constraints",
    "evaluate_surrogate",
    "generate_training_data",
    "generate_training_data_stratified",
    "load_surrogate_artifacts",
    "predict_price",
    "surrogate_ad_greeks",
    "surrogate_fd_greeks",
    "train_surrogate",
]

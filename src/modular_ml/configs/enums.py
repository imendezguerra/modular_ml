"""Enums used across the config dataclasses (replace bare string Literals)."""

from enum import Enum


class DType(str, Enum):
    float32 = "float32"
    float64 = "float64"


class Nonlinearity(str, Enum):
    tanh = "tanh"
    relu = "relu"
    none = "none"


class OptimizerType(str, Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    SGD = "SGD"


class TaskType(str, Enum):
    regression = "regression"
    classification = "classification"


class LossCriterion(str, Enum):
    mse = "mse"
    l1 = "l1"
    cross_entropy = "cross_entropy"
    norm2 = "norm2"
    pow2 = "pow2"
    var = "var"
    monotonicity = "monotonicity"
    trend = "trend"
    temporal_smoothness = "temporal_smoothness"
    effective_rank = "effective_rank"


class RunMode(str, Enum):
    train = "train"
    test = "test"
    train_test = "train_test"
    retrain = "retrain"
    retrain_test = "retrain_test"


class InitLoadMode(str, Enum):
    weights_only = "weights_only"
    full_state = "full_state"

"""Enums used across the config dataclasses (replace bare string Literals)."""

from enum import Enum


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
    huber = "huber"
    cross_entropy = "cross_entropy"
    bce_with_logits = "bce_with_logits"
    cosine = "cosine"
    norm1 = "norm1"
    norm2 = "norm2"
    norm_row = "norm_row"
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


class InitLoadMode(str, Enum):
    weights_only = "weights_only"
    full_state = "full_state"

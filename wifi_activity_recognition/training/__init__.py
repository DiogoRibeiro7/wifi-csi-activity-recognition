"""Training utilities package."""

from .callbacks import (
    Callback,
    EarlyStopping,
    LRScheduler,
    ModelCheckpoint,
    TensorBoardLogger,
)
from .federated import (
    FederatedClient,
    FederatedServer,
    add_gaussian_noise,
    clip_gradients,
    fed_avg,
    fed_prox,
    hardware_balanced_selector,
    random_selector,
    run_simulation,
    secure_aggregate,
)
from .losses import cross_entropy_loss, focal_loss, label_smoothing_loss
from .metrics import classification_metrics
from .trainer import Trainer

__all__ = [
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
    "TensorBoardLogger",
    "FederatedClient",
    "FederatedServer",
    "fed_avg",
    "fed_prox",
    "add_gaussian_noise",
    "clip_gradients",
    "secure_aggregate",
    "random_selector",
    "hardware_balanced_selector",
    "run_simulation",
    "cross_entropy_loss",
    "focal_loss",
    "label_smoothing_loss",
    "classification_metrics",
]

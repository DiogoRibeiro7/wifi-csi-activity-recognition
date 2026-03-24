"""Research-oriented extensions for WiFi activity recognition."""

from .domain_adaptation import (
    DomainAdapter,
    DomainAdversarialNetwork,
    coral_loss,
    grad_reverse,
    mmd_loss,
    wasserstein_distance,
)
from .few_shot_learning import (
    FewShotLearner,
    MAMLLearner,
    PrototypicalNetwork,
    RelationNetwork,
)

__all__ = [
    "DomainAdapter",
    "DomainAdversarialNetwork",
    "coral_loss",
    "mmd_loss",
    "wasserstein_distance",
    "grad_reverse",
    "FewShotLearner",
    "MAMLLearner",
    "PrototypicalNetwork",
    "RelationNetwork",
]

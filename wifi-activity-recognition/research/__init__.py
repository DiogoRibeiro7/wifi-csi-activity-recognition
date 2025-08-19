"""Research-oriented extensions for WiFi activity recognition."""

from .domain_adaptation import DomainAdapter
from .few_shot_learning import (
    FewShotLearner,
    MAMLLearner,
    PrototypicalNetwork,
    RelationNetwork,
)

__all__ = [
    "DomainAdapter",
    "FewShotLearner",
    "MAMLLearner",
    "PrototypicalNetwork",
    "RelationNetwork",
]

"""SAN and Furious: composable speckle-nulling forward models and algorithms.

Quick start::

    from san import CoronagraphModel, SANAndFurious

    model = CoronagraphModel()
    nuller = SANAndFurious(model)
    for _ in range(10):
        nuller.step()
        print(nuller.contrast)
"""

from .models import CoronagraphModel
from .algorithms import (
    SpeckleNuller,
    SpeckleAreaNulling,
    SANAndFurious,
    MinStepNulling,
    FastAndFurious,
    FastAndFuriousNoProbe,
)

__all__ = [
    "CoronagraphModel",
    "SpeckleNuller",
    "SpeckleAreaNulling",
    "SANAndFurious",
    "MinStepNulling",
    "FastAndFurious",
    "FastAndFuriousNoProbe",
]

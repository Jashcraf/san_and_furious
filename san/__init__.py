"""SAN and Furious: composable speckle-nulling forward models and algorithms.

Quick start::

    from san import CoronagraphModel, FastAndFuriousNoProbe

    model = CoronagraphModel()
    nuller = FastAndFuriousNoProbe(model)
    for _ in range(10):
        nuller.step()
        print(nuller.contrast)
"""

from .models import CoronagraphModel
from .algorithms import (
    SpeckleNuller,
    SpeckleAreaNulling,
    MinStepNulling,
    LagStepNulling,
)

__all__ = [
    "CoronagraphModel",
    "SpeckleNuller",
    "SpeckleAreaNulling",
    "MinStepNulling",
    "LagStepNulling",
]

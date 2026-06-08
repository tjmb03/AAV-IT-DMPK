"""
aav_it_dmpk - a mechanistic translational PK/biodistribution framework for
intrathecally delivered AAV gene therapy (mouse -> NHP -> human).

This is an illustrative scaffold for demonstrating a functional-modeling
approach. Parameters are representative literature values and must be replaced
with program-specific data before use in decision-making.
"""

from . import (physiology, model, translation, data_integration,
               biodistribution, pd_safety, validation, vpc, plotting)  # noqa
__all__ = ["physiology", "model", "translation", "data_integration",
           "biodistribution", "pd_safety", "validation", "vpc", "plotting"]

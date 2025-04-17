import numpy as np
from enum import Enum, unique

import torch


@unique
class Components(Enum):
    """
    Components of the action space.
    """
    IVF = 0
    Norepinephrine = 1
    Vasopressin = 2

    @staticmethod
    def n_components() -> int:
        return len(Components.__members__)


@unique
class IVF(Enum):
    """
    IVF: 25%: 10 ml, 50%: 29.2 ml, 75%: 100 ml
    """
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(value: float) -> 'IVF':
        return IVF.Yes if value > 0 else IVF.No


@unique
class Norepinephrine(Enum):
    """
    norepinephrine: 25%: 0.24 mcg/kg/min, 50%: 0.48 mcg/kg/min, 75%: 0.96 mcg/kg/min
    """
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(value: float) -> 'Norepinephrine':
        return Norepinephrine.Yes if value > 0 else Norepinephrine.No


@unique
class Vasopressin(Enum):
    """
    vasopressin: 25%: 0.02 units/min, 50%: 0.04 units/min, 75%: 0.08 units/min
    """
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(value: float) -> 'Vasopressin':
        return Vasopressin.Yes if value > 0 else Vasopressin.No

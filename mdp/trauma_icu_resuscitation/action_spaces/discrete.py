import numpy as np
from enum import Enum, unique


@unique
class Components(Enum):
    """
    NOTE: Dr. Beni suggested we only use the first three components for now.
    NOTE: Dr. OKeefe is going to look into the other components.
    NOTE: Drugs should be administered as follows: ivf -> norepinephrine -> vasopressin
    """
    IVF = 0
    Norepinephrine = 1
    Vasopressin = 2
    # Phenylephrine = 3
    # Dobutamine = 4
    # Dopamine = 5
    # Epinephrine = 6
    # Ephedrine = 7 # Dr. OKeefe says to ignore this one

    @staticmethod
    def n_components() -> int:
        return len(Components.__members__)


@unique
class IVF(Enum):
    """
    IVF: 250 ml (mild dose), 500 ml (slightly more aggressive dose)
    """
    NoDose = 0 # 0 ml <= ivf < 250 ml
    MildDose = 1 # 250 ml <= ivf < 500 ml
    HigherDose = 2 # ivf >= 500 ml

    @staticmethod
    def Discretize(value: float) -> 'IVF':
        match value:
            case _ as v if v < 250.0:
                return IVF.NoDose
            case _ as v if 250.0 <= v < 500 or np.isclose(v, 250.0):
                return IVF.MildDose
            case _ as v if v >= 500.0 or np.isclose(v, 500.0):
                return IVF.HigherDose


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


"""
IVF: 25%: 10 ml, 50%: 29.2 ml, 75%: 100 ml
- 0 ml: 0
- 0 ml < ivf <= 10 ml: 1
- 10 ml < ivf <= 29.2 ml: 2
- 29.2 ml < ivf <= 100 ml: 3
- ivf > 100 ml: 4

Vasopressors:
- norepinephrine: 25%: 0.24 mcg/kg/min, 50%: 0.48 mcg/kg/min, 75%: 0.96 mcg/kg/min
    - 0 mcg/kg/min: 0
    - 0 mcg/kg/min < dose <= 0.24 mcg/kg/min: 1
    - 0.24 mcg/kg/min < dose <= 0.48 mcg/kg/min: 2
    - 0.48 mcg/kg/min < dose <= 0.96 mcg/kg/min: 3
    - dose > 0.96 mcg/kg/min: 4
- vasopressin: 25%: 2.4 units/hr, 50%: 2.4 units/hr, 75%: 2.4 units/hr
    - 0 units/hr: 0
    - 0 units/hr < dose <= 1.0 units/hr: 1
    - 1.0 units/hr < dose <= 2.4 units/hr: 2
    - 2.4 units/hr < dose <= 5.0 units/hr: 3
    - dose > 5.0 units/hr: 4
- phenylephrine: 25%: 1.06 mcg/kg/min, 50%: 2.385 mcg/kg/min, 75%: 4.4832 mcg/kg/min
    - 0 mcg/kg/min: 0
    - 0 mcg/kg/min < dose <= 1.06 mcg/kg/min: 1
    - 1.06 mcg/kg/min < dose <= 2.385 mcg/kg/min: 2
    - 2.385 mcg/kg/min < dose <= 4.4832 mcg/kg/min: 3
    - dose > 4.4832 mcg/kg/min: 4
- DOBUTamine: 25%: 8.48 mcg/kg/min, 50%: 14.96 mcg/kg/min, 75%: 24.32 mcg/kg/min
    - 0 mcg/kg/min: 0
    - 0 mcg/kg/min < dose <= 8.48 mcg/kg/min: 1
    - 8.48 mcg/kg/min < dose <= 14.96 mcg/kg/min: 2
    - 14.96 mcg/kg/min < dose <= 24.32 mcg/kg/min: 3
    - dose > 24.32 mcg/kg/min: 4
- DOPamine: 25%: 14.592 mcg/kg/min, 50%: 22.816 mcg/kg/min, 75%: 36.448 mcg/kg/min
    - 0 mcg/kg/min: 0
    - 0 mcg/kg/min < dose <= 14.592 mcg/kg/min: 1
    - 14.592 mcg/kg/min < dose <= 22.816 mcg/kg/min: 2
    - 22.816 mcg/kg/min < dose <= 36.448 mcg/kg/min: 3
    - dose > 36.448 mcg/kg/min: 4
- EPINEPHrine: 25%: 0.318 mcg/kg/min, 50%: 0.626 mcg/kg/min, 75%: 1.5 mcg/kg/min
    - 0 mcg/kg/min: 0
    - 0 mcg/kg/min < dose <= 0.318 mcg/kg/min: 1
    - 0.318 mcg/kg/min < dose <= 0.626 mcg/kg/min: 2
    - 0.626 mcg/kg/min < dose <= 1.5 mcg/kg/min: 3
    - dose > 1.5 mcg/kg/min: 4
- ePHEDrine: 25%: 25.0 mg/hr, 50%: 25.0 mg/hr, 75%: 25.0 mg/hr
    - 0 mg/hr: 0
    - 0 mg/hr < dose <= 10 mg/hr: 1
    - 10 mg/hr < dose <= 25 mg/hr: 2
    - 25 mg/hr < dose <= 50 mg/hr: 3
    - dose > 50 mg/hr: 4
"""

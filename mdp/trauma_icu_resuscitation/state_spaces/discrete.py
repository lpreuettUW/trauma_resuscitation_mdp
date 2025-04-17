import numpy as np
from enum import Enum, unique
from typing import Type


@unique
class Components(Enum):
    # region Static Components
    Age = 0
    Sex = 1
    Height = 2
    Weight = 3
    # BMI = 2
    # Diabetes = 3
    # Smoker = 4
    HeadAIS = 4
    # ChestAIS = 6
    # AbdomenAIS = 7
    # SpineAIS = 8
    # LowerExtremityAIS = 9
    TraumaType = 5
    # Transferred = 11
    InitialRBC = 6
    InitialFFP = 7
    InitialPlatelets = 8
    InitialLactate = 9
    ICUAdmitLactate = 10
    VasopressorDoseWithinLastHour = 11
    # ORPriorToICU = 17 # idk which column to use for this
    # endregion

    # region Lab and Vital Components
    HR = 12
    SBP = 13
    DBP = 14
    # RR = 20
    MAP = 15
    # FiO2 = 22
    SpO2 = 16
    UOP = 17
    Temp = 18
    HCO3 = 19
    Lactate = 20
    Creatinine = 21
    pH = 22
    INR = 23
    # PaO2 = 31
    PaCO2 = 24
    Hgb = 25
    # endregion

    @staticmethod
    def n_static_components() -> int:
        return 12

    @staticmethod
    def n_lab_vital_components() -> int:
        return 14

    @staticmethod
    def n_components() -> int:
        return len(Components.__members__)

    @staticmethod
    def GetTypeFromComponent(component: Enum) -> Type:
        match component:
            case Components.Age:
                return Age
            case Components.Sex:
                return Sex
            case Components.Height:
                return Height
            case Components.Weight:
                return Weight
            # case Components.BMI:
            #     return BMI
            # case Components.Diabetes:
            #     return Diabetic
            # case Components.Smoker:
            #     return Smoker
            case Components.HeadAIS: # | Components.ChestAIS | Components.AbdomenAIS | Components.SpineAIS | Components.LowerExtremityAIS:
                return AIS
            case Components.TraumaType:
                return TraumaType
            # case Components.Transferred:
            #     return Transferred
            case Components.InitialRBC:
                return InitialRBC
            case Components.InitialFFP:
                return InitialFFP
            case Components.InitialPlatelets:
                return InitialPlatelets
            case Components.InitialLactate | Components.ICUAdmitLactate | Components.Lactate:
                return Lactate
            case Components.VasopressorDoseWithinLastHour:
                return VasopressorDoseWithinLastHour
            case Components.HR:
                return HR
            case Components.SBP:
                return SBP
            case Components.DBP:
                return DBP
            # case Components.RR:
            #     return RR
            case Components.MAP:
                return MAP
            # case Components.FiO2:
            #     return FiO2
            case Components.SpO2:
                return SpO2
            case Components.UOP:
                return UOP
            case Components.Temp:
                return Temp
            case Components.HCO3:
                return HCO3
            case Components.Creatinine:
                return Creatinine
            case Components.pH:
                return pH
            case Components.INR:
                return INR
            # case Components.PaO2:
            #     return PaO2
            case Components.PaCO2:
                return PaCO2
            case Components.Hgb:
                return Hgb

@unique
class Age(Enum):
    Adult = 0 # 18-39
    MiddleAged = 1 # 40-64
    Senior = 2 # 65-84
    Elderly = 3 # 85+

    @staticmethod
    def Discretize(age: int):
        match age:
            case _ as a if 0 <= a <= 17:
                raise ValueError('Age must be 18 or older')
            case _ as a if 18 <= a <= 39:
                return Age.Adult
            case _ as a if 40 <= a <= 64:
                return Age.MiddleAged
            case _ as a if 65 <= a <= 84:
                return Age.Senior
            case _:
                return Age.Elderly

@unique
class Sex(Enum):
    Female = 0
    Male = 1

    @staticmethod
    def Discretize(male: int) -> 'Sex':
        return Sex.Male if male else Sex.Female

@unique
class Height(Enum):
    Unknown = 0
    VeryShort = 1 # height < 147
    Short = 2 # 147 <= height < 168
    Normal = 3 # 168 <= height <= 182
    Tall = 4 # height > 182

    @staticmethod
    def Discretize(height: float) -> 'Height':
        match height:
            case _ as h if np.isnan(h):
                return Height.Unknown
            case _ as h if h < 147:
                return Height.VeryShort
            case _ as h if h < 168:
                return Height.Short
            case _ as h if h <= 182 or np.isclose(height, 182.0):
                return Height.Normal
            case _:
                return Height.Tall

@unique
class Weight(Enum):
    Unknown = 0
    VeryLow = 1 # weight < 32.5
    Low = 2 # 32.5 <= weight < 70
    Normal = 3 # 70 <= weight <= 95
    High = 4 # 95 < weight <= 132.5
    VeryHigh = 5 # weight > 132.5

    @staticmethod
    def Discretize(weight: float) -> 'Weight':
        match weight:
            case _ as w if np.isnan(w):
                return Weight.Unknown
            case _ as w if w < 32.5:
                return Weight.VeryLow
            case _ as w if w < 70:
                return Weight.Low
            case _ as w if w <= 95 or np.isclose(weight, 95.0):
                return Weight.Normal
            case _ as w if w <= 132.5 or np.isclose(weight, 132.5):
                return Weight.High
            case _:
                return Weight.VeryHigh

@unique
class BMI(Enum):
    Unknown = 0 # BMI = NaN
    Underweight = 1 # BMI < 18.5
    Normal = 2 # 18.5 <= BMI < 25
    Overweight = 3 # 25 <= BMI < 30
    Obese_Class_I = 4 # 30 <= BMI < 35
    Obese_Class_II = 5 # 35 <= BMI < 40
    Obese_Class_III = 6 # BMI >= 40

    @staticmethod
    def Discretize(bmi: float) -> 'BMI':
        match bmi:
            case _ as b if np.isnan(b):
                return BMI.Unknown
            case _ as b if b < 18.5:
                return BMI.Underweight
            case _ as b if b < 25:
                return BMI.Normal
            case _ as b if b < 30:
                return BMI.Overweight
            case _ as b if b < 35:
                return BMI.Obese_Class_I
            case _ as b if b < 40:
                return BMI.Obese_Class_II
            case _:
                return BMI.Obese_Class_III

@unique
class Diabetic(Enum):
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(diabetic: int) -> 'Diabetic':
        return Diabetic.Yes if diabetic else Diabetic.No

@unique
class Smoker(Enum):
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(smoker: int) -> 'Smoker':
        return Smoker.Yes if smoker else Smoker.No

@unique
class AIS(Enum): # Head, Chest, Abdomen, Spine, Lower Extremity
    LowSeverity = 0 # 1-2
    Severe = 1 # 3-6

    @staticmethod
    def Discretize(ais: int) -> 'AIS':
        match ais:
            case _ as a if a < 3:
                return AIS.LowSeverity
            case _:
                return AIS.Severe

@unique
class TraumaType(Enum):
    Blunt = 0
    Penetrating = 1
    Other = 2

    @staticmethod
    def Discretize(trauma_type: str) -> 'TraumaType':
        match trauma_type.strip():
            case 'B':
                return TraumaType.Blunt
            case 'P':
                return TraumaType.Penetrating
            case 'OTHER':
                return TraumaType.Other
            case _:
                raise ValueError(f'Invalid trauma type: {trauma_type}')

@unique
class Transferred(Enum):
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(transferred: int) -> 'Transferred':
        return Transferred.Yes if transferred else Transferred.No

# @unique # idk which column to use for this
# class ORPriorToICU(Enum):
#     No = 0
#     Yes = 1

@unique
class Lactate(Enum): # Initial Lacate, ICU Admit Lactate, and Lactate (Lab)
    Unknown = 0 # lactate = NaN
    Normal = 1 # lactate <= 2.0
    SlightlyElevated = 2 # 2.0 < lactate < 5.0
    Elevated = 3 # 5.0 <= lactate < 10.0
    VeryElevated = 4 # lactate >= 10.0

    @staticmethod
    def Discretize(lactate: float) -> 'Lactate':
        match lactate:
            case _ as l if np.isnan(l):
                return Lactate.Unknown
            case _ as l if l <= 2.0 or np.isclose(lactate, 2.0):
                return Lactate.Normal
            case _ as l if l < 5.0:
                return Lactate.SlightlyElevated
            case _ as l if l < 10.0:
                return Lactate.Elevated
            case _:
                return Lactate.VeryElevated

@unique
class InitialRBC(Enum): # units given
    None_Given = 0 # units = 0
    Few = 1 # 0 < units < 5
    Some = 2 # 5 <= units < 10
    Several = 3 # 10 <= units < 20
    Many = 4 # units >= 20

    @staticmethod
    def Discretize(rbc: float) -> 'InitialRBC':
        match rbc:
            case _ as r if r == 0 or np.isclose(rbc, 0):
                return InitialRBC.None_Given
            case _ as r if 0 < r < 5:
                return InitialRBC.Few
            case _ as r if 5 <= r < 10 or np.isclose(rbc, 5.0):
                return InitialRBC.Some
            case _ as r if 10 <= r < 20 or np.isclose(rbc, 10.0):
                return InitialRBC.Several
            case _:
                return InitialRBC.Many

@unique
class InitialFFP(Enum): # units given
    None_Given = 0 # units = 0
    Few = 1  # 0 < units < 5
    Some = 2  # 5 <= units < 10
    Several = 3  # 10 <= units < 20
    Many = 4  # units >= 20

    @staticmethod
    def Discretize(ffp: float) -> 'InitialFFP':
        match ffp:
            case _ as f if f == 0 or np.isclose(ffp, 0):
                return InitialFFP.None_Given
            case _ as f if 0 < f < 5:
                return InitialFFP.Few
            case _ as f if 5 <= f < 10 or np.isclose(ffp, 5.0):
                return InitialFFP.Some
            case _ as f if 10 <= f < 20 or np.isclose(ffp, 10.0):
                return InitialFFP.Several
            case _:
                return InitialFFP.Many

@unique
class InitialPlatelets(Enum): # units given
    None_Given = 0 # units = 0
    Few = 1 # 0 < units < 2
    Some = 2 # 2 <= units < 6
    Many = 3 # units >= 6

    @staticmethod
    def Discretize(platelets: float) -> 'InitialPlatelets':
        match platelets:
            case _ as p if p == 0 or np.isclose(platelets, 0):
                return InitialPlatelets.None_Given
            case _ as p if 0 < p < 2:
                return InitialPlatelets.Few
            case _ as p if 2 <= p < 6 or np.isclose(platelets, 2.0):
                return InitialPlatelets.Some
            case _:
                return InitialPlatelets.Many

@unique
class UOP(Enum):
    Unknown = 0 # UOP = NaN
    Low = 1 # UOP < 0.5
    Normal = 2 # 0.5 <= UOP <= 1.5
    High = 3 # UOP > 1.5

    @staticmethod
    def Discretize(uop: float) -> 'UOP':
        match uop:
            case _ as u if np.isnan(u):
                return UOP.Unknown
            case _ as u if u < 0.5:
                return UOP.Low
            case _ as u if u <= 1.5 or np.isclose(uop, 1.5):
                return UOP.Normal
            case _:
                return UOP.High

@unique
class HCO3(Enum):
    Unknown = 0 # HCO3 = NaN
    VeryLow = 1 # 0 <= HCO3 < 11
    Low = 2 # 11 <= HCO3 < 22
    Normal = 3 # 22 <= HCO3 <= 29
    High = 4 # 29 < HCO3 <= 40
    VeryHigh = 5 # HCO3 > 40

    @staticmethod
    def Discretize(hco3: float) -> 'HCO3':
        match hco3:
            case _ as h if np.isnan(h):
                return HCO3.Unknown
            case _ as h if h < 11:
                return HCO3.VeryLow
            case _ as h if h < 22:
                return HCO3.Low
            case _ as h if h <= 29 or np.isclose(hco3, 29.0):
                return HCO3.Normal
            case _ as h if h <= 40 or np.isclose(hco3, 40.0):
                return HCO3.High
            case _:
                return HCO3.VeryHigh

@unique
class VasopressorDoseWithinLastHour(Enum):
    No = 0
    Yes = 1

    @staticmethod
    def Discretize(flag: int) -> 'VasopressorDoseWithinLastHour':
        return VasopressorDoseWithinLastHour.Yes if flag > 0 else VasopressorDoseWithinLastHour.No

class HR: # discrete values
    Min = 0
    Max = 250

    @staticmethod
    def Discretize(hr: int) -> int:
        # NOTE: we shift by one because 0 is reserved for unknown values
        match hr:
            case _ as h if np.isnan(h):
                return 0
            case _ as h if h < HR.Min:
                return HR.Min + 1
            case _ as h if h > HR.Max:
                return HR.Max + 1
            case _:
                return hr + 1

class SBP: # discrete values
    Min = 0
    Max = 300

    @staticmethod
    def Discretize(sbp: int) -> int:
        # NOTE: we shift by one because 0 is reserved for unknown values
        match sbp:
            case _ as s if np.isnan(s):
                return 0
            case _ as s if s < SBP.Min:
                return SBP.Min + 1
            case _ as s if s > SBP.Max:
                return SBP.Max + 1
            case _:
                return sbp + 1

class DBP: # discrete values
    Min = 0
    Max = 250

    @staticmethod
    def Discretize(dbp: int) -> int:
        # NOTE: we shift by one because 0 is reserved for unknown values
        match dbp:
            case _ as d if np.isnan(d):
                return 0
            case _ as d if d < DBP.Min:
                return DBP.Min + 1
            case _ as d if d > DBP.Max:
                return DBP.Max + 1
            case _:
                return dbp + 1

class MAP: # discrete values
    Min = 0
    Max = 250

    @staticmethod
    def Discretize(map: int) -> int:
        # NOTE: we shift by one because 0 is reserved for unknown values
        match map:
            case _ as m if np.isnan(m):
                return 0
            case _ as m if m < MAP.Min:
                return MAP.Min + 1
            case _ as m if m > MAP.Max:
                return MAP.Max + 1
            case _:
                return map + 1

@unique
class RR(Enum):
    Unknown = 0 # RR = NaN
    VeryLow = 1 # 0 <= RR <= 5
    Low = 2 # 5 < RR <= 11
    Normal = 3 # 11 < RR <= 18
    High = 4 # 18 < RR <= 30
    VeryHigh = 5 # RR > 30

    @staticmethod
    def Discretize(rr: int) -> 'RR':
        match rr:
            case _ as r if np.isnan(r):
                return RR.Unknown
            case _ as r if r <= 5:
                return RR.VeryLow
            case _ as r if r <= 11:
                return RR.Low
            case _ as r if r <= 18:
                return RR.Normal
            case _ as r if r <= 30:
                return RR.High
            case _:
                return RR.VeryHigh

@unique
class FiO2(Enum):
    Unknown = 0 # FiO2 = NaN
    VeryLow = 1 # 0 <= FiO2 <= 19
    Low = 2 # 19 < FiO2 < 35
    Normal = 3 # 35 <= FiO2 <= 50
    High = 4 # 50 < FiO2 <= 75
    VeryHigh = 5 # FiO2 > 75

    @staticmethod
    def Discretize(fio2: int) -> 'FiO2':
        match fio2:
            case _ as f if np.isnan(f):
                return FiO2.Unknown
            case _ as f if f <= 19:
                return FiO2.VeryLow
            case _ as f if f < 35:
                return FiO2.Low
            case _ as f if f <= 50:
                return FiO2.Normal
            case _ as f if f <= 75:
                return FiO2.High
            case _:
                return FiO2.VeryHigh

@unique
class SpO2(Enum):
    Unknown = 0 # SpO2 = NaN
    Critical = 1 # 0 < SpO2 < 20
    VeryLow = 2 # 20 <= SpO2 <= 50
    Low = 3 # 50 < SpO2 <= 80
    SlightlyLow = 4 # 80 < SpO2 < 92
    Normal = 5 # 92 <= SpO2 <= 100

    @staticmethod
    def Discretize(spo2: int) -> 'SpO2':
        match spo2:
            case _ as s if np.isnan(s):
                return SpO2.Unknown
            case _ as s if 0 < s < 20:
                return SpO2.Critical
            case _ as s if s <= 50:
                return SpO2.VeryLow
            case _ as s if s <= 80:
                return SpO2.Low
            case _ as s if s < 92:
                return SpO2.SlightlyLow
            case _:
                return SpO2.Normal

@unique
class Temp(Enum):
    Unknown = 0 # Temp = NaN
    SevereHypothermia = 1 # Temp < 33
    ModerateHypothermia = 2 # 33 <= Temp <= 34.5
    MildHypothermia = 3 # 34.5 < Temp <= 37
    Normal = 4 # 37 < Temp <= 38
    Fever = 5 # Temp > 38

    @staticmethod
    def Discretize(temp: float) -> 'Temp':
        match temp:
            case _ as t if np.isnan(t):
                return Temp.Unknown
            case _ as t if t < 33:
                return Temp.SevereHypothermia
            case _ as t if t <= 34.5 or np.isclose(temp, 34.5):
                return Temp.ModerateHypothermia
            case _ as t if t <= 37 or np.isclose(temp, 37.0):
                return Temp.MildHypothermia
            case _ as t if t <= 38 or np.isclose(temp, 38.0):
                return Temp.Normal
            case _:
                return Temp.Fever

@unique
class Hgb(Enum):
    Unknown = 0 # hgb = NaN
    VeryLow = 1 # male hgb < 13, female hgb < 12
    Low = 2 # male 13 <= hgb < 14, female 12 <= hgb < 12.3
    Normal = 3 # male 14 <= hgb <= 17.5, female 12.3 <= hgb <= 15.3
    High = 4 # male 17.5 < hgb <= 18, female 15.3 < hgb <= 16
    VeryHigh = 5 # male hgb > 18, female hgb > 16

    @staticmethod
    def Discretize(hgb: float, male: int) -> 'Hgb':
        if male:
            match hgb:
                case _ as h if np.isnan(h):
                    return Hgb.Unknown
                case _ as h if h < 13:
                    return Hgb.VeryLow
                case _ as h if h < 14:
                    return Hgb.Low
                case _ as h if h <= 17.5 or np.isclose(hgb, 17.5):
                    return Hgb.Normal
                case _ as h if h <= 18 or np.isclose(hgb, 18.0):
                    return Hgb.High
                case _:
                    return Hgb.VeryHigh
        else: # female
            match hgb:
                case _ as h if np.isnan(h):
                    return Hgb.Unknown
                case _ as h if h < 12:
                    return Hgb.VeryLow
                case _ as h if h < 12.3:
                    return Hgb.Low
                case _ as h if h <= 15.3 or np.isclose(hgb, 15.3):
                    return Hgb.Normal
                case _ as h if h <= 16 or np.isclose(hgb, 16.0):
                    return Hgb.High
                case _:
                    return Hgb.VeryHigh

@unique
class PaO2(Enum):
    Unknown = 0 # PaO2 = NaN
    VeryLow = 1 # PaO2 <= 50
    Low = 2 # 50 < PaO2 < 75
    Normal = 3 # 75 <= PaO2 <= 100
    High = 4 # 100 < PaO2 <= 125
    VeryHigh = 5 # PaO2 > 125

    @staticmethod
    def Discretize(pao2: int) -> 'PaO2':
        match pao2:
            case _ as p if np.isnan(p):
                return PaO2.Unknown
            case _ as p if p <= 50:
                return PaO2.VeryLow
            case _ as p if p < 75:
                return PaO2.Low
            case _ as p if p <= 100:
                return PaO2.Normal
            case _ as p if p <= 125:
                return PaO2.High
            case _:
                return PaO2.VeryHigh

@unique
class PaCO2(Enum):
    Unknown = 0
    VeryLow = 1 # PaCO2 <= 20
    Low = 2 # 20 < PaCO2 < 35
    Normal = 3 # 35 <= PaCO2 <= 45
    High = 4 # 45 < PaCO2 <= 60
    VeryHigh = 5 # PaCO2 > 60

    @staticmethod
    def Discretize(paco2: int) -> 'PaCO2':
        match paco2:
            case _ as p if np.isnan(p):
                return PaCO2.Unknown
            case _ as p if p <= 20:
                return PaCO2.VeryLow
            case _ as p if p < 35:
                return PaCO2.Low
            case _ as p if p <= 45:
                return PaCO2.Normal
            case _ as p if p <= 60:
                return PaCO2.High
            case _:
                return PaCO2.VeryHigh

@unique
class Creatinine(Enum):
    Unknown = 0 # creatinine = NaN
    Low = 1 # male creatinine < 0.7, female creatinine < 0.5
    Normal = 2 # male 0.7 <= creatinine <= 1.3, female 0.5 <= creatinine <= 1.1
    High = 3 # male 1.3 < creatinine, female 1.1 < creatinine

    @staticmethod
    def Discretize(creatinine: float, male: int) -> 'Creatinine':
        if male:
            match creatinine:
                case _ as c if np.isnan(c):
                    return Creatinine.Unknown
                case _ as c if c < 0.7:
                    return Creatinine.Low
                case _ as c if c <= 1.3 or np.isclose(creatinine, 1.3):
                    return Creatinine.Normal
                case _:
                    return Creatinine.High
        else:
            match creatinine:
                case _ as c if np.isnan(c):
                    return Creatinine.Unknown
                case _ as c if c < 0.5:
                    return Creatinine.Low
                case _ as c if c <= 1.1 or np.isclose(creatinine, 1.1):
                    return Creatinine.Normal
                case _:
                    return Creatinine.High

@unique
class pH(Enum):
    Unknown = 0 # pH = NaN
    VeryLow = 1 # pH < 7.0
    Low = 2 # 7.0 <= pH < 7.35
    Normal = 3 # 7.35 <= pH <= 7.45
    High = 4 # 7.45 < pH <= 7.6
    VeryHigh = 5 # pH > 7.6

    @staticmethod
    def Discretize(ph: float) -> 'pH':
        match ph:
            case _ as p if np.isnan(p):
                return pH.Unknown
            case _ as p if p < 7.0:
                return pH.VeryLow
            case _ as p if p < 7.35:
                return pH.Low
            case _ as p if p <= 7.45 or np.isclose(ph, 7.45):
                return pH.Normal
            case _ as p if p <= 7.6 or np.isclose(ph, 7.6):
                return pH.High
            case _:
                return pH.VeryHigh

@unique
class INR(Enum):
    Unknown = 0
    Normal = 1 # INR <= 1.1
    SlightlyElevated = 2 # 1.1 < INR <= 1.5
    High = 3 # 1.5 < INR <= 3.0
    Critical = 4 # INR > 3.0

    @staticmethod
    def Discretize(inr: float) -> 'INR':
        match inr:
            case _ as i if np.isnan(i):
                return INR.Unknown
            case _ as i if i <= 1.1 or np.isclose(inr, 1.1):
                return INR.Normal
            case _ as i if i <= 1.5 or np.isclose(inr, 1.5):
                return INR.SlightlyElevated
            case _ as i if i <= 3.0 or np.isclose(inr, 3.0):
                return INR.High
            case _:
                return INR.Critical

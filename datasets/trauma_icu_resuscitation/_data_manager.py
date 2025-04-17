import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Callable


class DataManager:
    """
    Class to manage datafile loading and converting to a unified format.
    """

    # initial lactate is lactate prior to ICU while lactate_value_ICU is lactate value at ICU admission
    _Static_Patient_Columns = [
        'StudyID', 'Height', 'Weight', 'Age', 'Male', # 'Diabetes', 'Smoker', 'MaxChestAIS', 'MaxAbdAIS',
        # 'MaxSpineAIS', 'MaxLEAIS', 'Transfer',
        'Pressors_ICU',
        'MaxHeadAIS', 'TraumaType', 'InitLactate', 'RBC_pre_ICU',
        'FFP_pre_ICU', 'PLT_pre_ICU', 'lactate_value_ICU',
        'ICU_AdmitDtTm', 'Deceased', 'VentFreeDays'
    ]

    _Labs_And_Vitals_Filenames = {
        'creatinine': {'MASTER_2012-2015_Creatinine.csv', 'MASTER_2016-2019_Creatinine.csv'},
        'inr': {'MASTER_2012-2015_INR.csv', 'MASTER_2016-2019_INR.csv'},
        'ph': {'MASTER_2012-2015_pH.csv', 'MASTER_2016-2019_pH.csv'},
        # 'fio2': {'2012-2015_Corrected_FiO2.csv', '2016-2019_Corrected_FiO2.csv'},
        # 'rr': {'MASTER_2012-2015_RR.csv', 'MASTER_2016-2019_RR.csv'},
        'map': {'MASTER_2012-2015_MAP.csv', 'MASTER_2016-2019_MAP.csv'},
        'hr': {'MASTER_2012-2015_HR.csv', 'MASTER_2016-2019_HR.csv'},
        'sbp': {'MASTER_2012-2015_SBP.csv', 'MASTER_2016-2019_SBP.csv'},
        'dbp': {'MASTER_2012-2015_DBP.csv', 'MASTER_2016-2019_DBP.csv'},
        'temp': {'MASTER_2012-2015_Temp.csv', 'MASTER_2016-2019_Temp.csv'},
        'uop': {'MASTER_2012-2015_Hourly_UOP_Total.csv', 'MASTER_2016-2019_UOP_no_dupes.csv'},
        'lactate': {'MASTER_2012-2015_lactate.csv', 'MASTER_2016-2019_lactate.csv'},
        # 'pao2': {'MASTER_2012-2015_PaO2.csv', 'MASTER_2016-2019_PaO2.csv'},
        'paco2': {'MASTER_2012-2015_PaCO2.csv', 'MASTER_2016-2019_PaCO2.csv'},
        'spo2': {'MASTER_2012-2015_SpO2.csv', 'MASTER_2016-2019_SpO2.csv'},
        'hgb': {'MASTER_2012-2015_hgb.csv', 'MASTER_2016-2019_hgb.csv'},
        'hco3': {'MASTER_2012-2015_HCO3.csv', 'MASTER_2016-2019_HCO3.csv'},
    }

    _Labs_And_Vitals_Column_Rename_Dict = { # StudyID, Observation, ObservationDtTm
        'MASTER_2012-2015_Creatinine.csv': {'ObservationValue': 'Observation'},
        'MASTER_2016-2019_Creatinine.csv': {'ObservationValue': 'Observation'},
        'MASTER_2012-2015_INR.csv': {'ObservationValue': 'Observation'},
        'MASTER_2016-2019_INR.csv': {'ObservationValue': 'Observation'},
        'MASTER_2012-2015_pH.csv': {'ObservationValue': 'Observation'},
        'MASTER_2016-2019_pH.csv': {'ObservationValue': 'Observation'},
        # '2012-2015_Corrected_FiO2.csv': {'FiO2': 'Observation'},
        # '2016-2019_Corrected_FiO2.csv': {'FiO2': 'Observation'},
        # 'MASTER_2012-2015_RR.csv': {'RESULT_VAL': 'Observation', 'STUDYID': 'StudyID'},
        # 'MASTER_2016-2019_RR.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_MAP.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2016-2019_MAP.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_HR.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2016-2019_HR.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_SBP.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2016-2019_SBP.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_DBP.csv': {'RESULT_VAL': 'Observation', 'STUDYID': 'StudyID'},
        'MASTER_2016-2019_DBP.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_Temp.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2016-2019_Temp.csv': {'Value': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_Hourly_UOP_Total.csv': {'Volume': 'Observation'},
        'MASTER_2016-2019_UOP_no_dupes.csv': {'Volume': 'Observation', 'EventEndDtTm': 'ObservationDtTm'},
        'MASTER_2012-2015_lactate.csv': {'ObservationValue': 'Observation'},
        'MASTER_2016-2019_lactate.csv': {'ObservationValue': 'Observation'},
        # 'MASTER_2012-2015_PaO2.csv': {'ObservationValue': 'Observation'},
        # 'MASTER_2016-2019_PaO2.csv': {'ObservationValue': 'Observation'},
        'MASTER_2012-2015_PaCO2.csv': {'ObservationValue': 'Observation'},
        'MASTER_2016-2019_PaCO2.csv': {'ObservationValue': 'Observation'},
        'MASTER_2012-2015_SpO2.csv': {'RESULT_VAL': 'Observation', 'STUDYID': 'StudyID'},
        'MASTER_2016-2019_SpO2.csv': {'SpO2': 'Observation', 'EVENT_END_DT_TM': 'ObservationDtTm', 'STUDYID': 'StudyID'},
        'MASTER_2012-2015_hgb.csv': {'STUDYID': 'StudyID', 'ObservationValue': 'Observation'},
        'MASTER_2016-2019_hgb.csv': {'STUDYID': 'StudyID', 'Value': 'Observation'},
        'MASTER_2012-2015_HCO3.csv': {'Value': 'Observation', 'STUDYID': 'StudyID'},
        'MASTER_2016-2019_HCO3.csv': {'Value': 'Observation', 'STUDYID': 'StudyID'},
    }

    _Labs_And_Vitals_Time_Conversion_Type_Dict = {
        'MASTER_2012-2015_Creatinine.csv': 'datetime',
        'MASTER_2016-2019_Creatinine.csv': 'datetime',
        'MASTER_2012-2015_INR.csv': 'datetime',
        'MASTER_2016-2019_INR.csv': 'datetime',
        'MASTER_2012-2015_pH.csv': 'datetime',
        'MASTER_2016-2019_pH.csv': 'datetime',
        # '2012-2015_Corrected_FiO2.csv': 'datetime',
        # '2016-2019_Corrected_FiO2.csv': 'datetime',
        # 'MASTER_2012-2015_RR.csv': ('day_time', ('DAY', 'TIME')),
        # 'MASTER_2016-2019_RR.csv': 'datetime',
        'MASTER_2012-2015_MAP.csv': 'datetime',
        'MASTER_2016-2019_MAP.csv': 'datetime',
        'MASTER_2012-2015_HR.csv': 'datetime',
        'MASTER_2016-2019_HR.csv': 'datetime',
        'MASTER_2012-2015_SBP.csv': 'datetime',
        'MASTER_2016-2019_SBP.csv': 'datetime',
        'MASTER_2012-2015_DBP.csv': ('day_time', ('DAY', 'TIME')),
        'MASTER_2016-2019_DBP.csv': 'datetime',
        'MASTER_2012-2015_Temp.csv': 'datetime',
        'MASTER_2016-2019_Temp.csv': 'datetime',
        'MASTER_2012-2015_Hourly_UOP_Total.csv': ('hours_from_admit', 'Hours_from_ICU_Admit'),
        'MASTER_2016-2019_UOP_no_dupes.csv': 'datetime',
        'MASTER_2012-2015_lactate.csv': 'datetime',
        'MASTER_2016-2019_lactate.csv': 'datetime',
        # 'MASTER_2012-2015_PaO2.csv': 'datetime',
        # 'MASTER_2016-2019_PaO2.csv': 'datetime',
        'MASTER_2012-2015_PaCO2.csv': 'datetime',
        'MASTER_2016-2019_PaCO2.csv': 'datetime',
        'MASTER_2012-2015_SpO2.csv': ('day_time', ('DAY', 'TIME')),
        'MASTER_2016-2019_SpO2.csv': 'datetime',
        'MASTER_2012-2015_hgb.csv': 'datetime',
        'MASTER_2016-2019_hgb.csv': 'datetime',
        'MASTER_2012-2015_HCO3.csv': ('day_time', ('DAY', 'TIME')),
        'MASTER_2016-2019_HCO3.csv': 'datetime',
    }

    _Labs_And_Vitals_Datetime_Format_Dict = {
        'MASTER_2012-2015_Creatinine.csv': '%m/%d/%y %H:%M',
        'MASTER_2016-2019_Creatinine.csv': '%m/%d/%y %H:%M',
        'MASTER_2012-2015_INR.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2016-2019_INR.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2012-2015_pH.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2016-2019_pH.csv': '%m/%d/%y %I:%M %p',
        # '2012-2015_Corrected_FiO2.csv': '%Y-%m-%d %H:%M:%S',
        # '2016-2019_Corrected_FiO2.csv': '%Y-%m-%d %H:%M:%S',
        # 'MASTER_2012-2015_RR.csv': '%H:%M:%S.%f',
        # 'MASTER_2016-2019_RR.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_MAP.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2016-2019_MAP.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_HR.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2016-2019_HR.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_SBP.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2016-2019_SBP.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_DBP.csv': '%I:%M:%S %p',
        'MASTER_2016-2019_DBP.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_Temp.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2016-2019_Temp.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_Hourly_UOP_Total.csv': None,
        'MASTER_2016-2019_UOP_no_dupes.csv': '%Y-%m-%d %H:%M:%S',
        'MASTER_2012-2015_lactate.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2016-2019_lactate.csv': '%m/%d/%y %I:%M %p',
        # 'MASTER_2012-2015_PaO2.csv': '%m/%d/%y %I:%M %p',
        # 'MASTER_2016-2019_PaO2.csv': '%m/%d/%y %H:%M',
        'MASTER_2012-2015_PaCO2.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2016-2019_PaCO2.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2012-2015_SpO2.csv': '%I:%M:%S %p',
        'MASTER_2016-2019_SpO2.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2012-2015_hgb.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2016-2019_hgb.csv': '%m/%d/%y %I:%M %p',
        'MASTER_2012-2015_HCO3.csv': '%H:%M:%S.%f',
        'MASTER_2016-2019_HCO3.csv': '%m/%d/%y %I:%M %p',
    }

    _Labs_And_Vitals_Obs_Vals_Conversion_Dict = {
        'MASTER_2012-2015_Creatinine.csv': 'coerce',
        'MASTER_2016-2019_Creatinine.csv': None,
        'MASTER_2012-2015_INR.csv': None,
        'MASTER_2016-2019_INR.csv': None,
        'MASTER_2012-2015_pH.csv': None,
        'MASTER_2016-2019_pH.csv': None,
        # '2012-2015_Corrected_FiO2.csv': None,
        # '2016-2019_Corrected_FiO2.csv': None,
        # 'MASTER_2012-2015_RR.csv': None,
        # 'MASTER_2016-2019_RR.csv': None,
        'MASTER_2012-2015_MAP.csv': None,
        'MASTER_2016-2019_MAP.csv': None,
        'MASTER_2012-2015_HR.csv': None,
        'MASTER_2016-2019_HR.csv': None,
        'MASTER_2012-2015_SBP.csv': None,
        'MASTER_2016-2019_SBP.csv': None,
        'MASTER_2012-2015_DBP.csv': None,
        'MASTER_2016-2019_DBP.csv': None,
        'MASTER_2012-2015_Temp.csv': None,
        'MASTER_2016-2019_Temp.csv': None,
        'MASTER_2012-2015_Hourly_UOP_Total.csv': None,
        'MASTER_2016-2019_UOP_no_dupes.csv': None,
        'MASTER_2012-2015_lactate.csv': None,
        'MASTER_2016-2019_lactate.csv': None,
        # 'MASTER_2012-2015_PaO2.csv': 'cast',
        # 'MASTER_2016-2019_PaO2.csv': 'cast',
        'MASTER_2012-2015_PaCO2.csv': 'cast',
        'MASTER_2016-2019_PaCO2.csv': 'cast',
        'MASTER_2012-2015_SpO2.csv': 'coerce',
        'MASTER_2016-2019_SpO2.csv': 'coerce',
        'MASTER_2012-2015_hgb.csv': None,
        'MASTER_2016-2019_hgb.csv': None,
        'MASTER_2012-2015_HCO3.csv': 'coerce',
        'MASTER_2016-2019_HCO3.csv': 'cast',
    }

    _IVF_Column_Rename_Dict = {
        '2012-2015': {'EventEndDtTm': 'ObservationDtTm', 'Volume': 'Observation', 'UNIT': 'Unit', 'FLUID': 'Type', 'ROUTE': 'Route'},
        '2016-2019': {'EventEndDtTm': 'ObservationDtTm', 'Volume': 'Observation', 'RESULT_UNITS_CD_DESCR': 'Unit', 'EVENT_CD_DESCR': 'Type', 'OE_FIELD_DISPLAY_VALUE': 'Route'},
    }

    _IVF_Drop_Columns = {
        '2012-2015': ['ADMITDATE', 'DAY'],
        '2016-2019': ['AdmitDate', 'Unnamed: 3'],
    }

    def __init__(self, base_path: str, cohort_selector: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, hours_preceding_icu_admit: int = 24, hours_post_icu_admit: int = 48,
                 observation_smoother: Optional[Callable[[pd.DataFrame, str, str], pd.DataFrame]] = None):
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base path {base_path} does not exist.")
        self._hours_preceding_icu_admit = hours_preceding_icu_admit
        self._hours_post_icu_admit = hours_post_icu_admit
        # load patient data
        patient_df_0 = pd.read_csv(os.path.join(base_path, '2012-2015_compiled_patient_data_with_cohorts_no_L_required_new_IVF_EarlyLateBolus.csv'))
        patient_df_1 = pd.read_csv(os.path.join(base_path, '2016-2019_compiled_patient_data_with_cohorts_no_L_required_new_IVF_EarlyLateBolus.csv'))
        patient_df_1.rename(columns={'Pregancy': 'Pregnancy'}, inplace=True)
        # cast studyid
        patient_df_0['StudyID'] = patient_df_0['StudyID'].astype(int)
        patient_df_1['StudyID'] = patient_df_1['StudyID'].astype(int)
        if cohort_selector is not None:
            patient_df_0 = cohort_selector(patient_df_0)
            patient_df_1 = cohort_selector(patient_df_1)
        # select subset of columns
        patient_df_0 = patient_df_0[self._Static_Patient_Columns]
        patient_df_1 = patient_df_1[self._Static_Patient_Columns]
        # merge patient data
        self._patient_df = pd.concat([patient_df_0, patient_df_1], ignore_index=True)
        # parse icu admit datetime
        self._patient_df['ICU_AdmitDtTm'] = pd.to_datetime(self._patient_df['ICU_AdmitDtTm'], format='%Y-%m-%d %H:%M:%S')
        self._patient_df['Height'] = pd.to_numeric(self._patient_df['Height'], errors='coerce')
        self._patient_df['Weight'] = pd.to_numeric(self._patient_df['Weight'], errors='coerce')
        # load labs and vitals data
        self._labs_vitals_df_dict = dict()
        for key, filenames in self._Labs_And_Vitals_Filenames.items():
            lab_vital_df_list = list()
            for filename in filenames:
                try:
                    lab_vital_df = pd.read_csv(os.path.join(base_path, filename))
                    rename_dict = self._Labs_And_Vitals_Column_Rename_Dict[filename]
                    # ensure we arent duplicating an existing column name
                    cols_to_drop = set(lab_vital_df.columns) & set(rename_dict.values())
                    if 'ICU_AdmitDtTm' in lab_vital_df.columns:
                        cols_to_drop.add('ICU_AdmitDtTm')
                    if cols_to_drop:
                        lab_vital_df.drop(columns=list(cols_to_drop), inplace=True)
                    lab_vital_df.rename(columns=rename_dict, inplace=True)
                    # filter out patients not in our patient cohort
                    lab_vital_df = lab_vital_df[lab_vital_df['StudyID'].isin(self._patient_df['StudyID'])]
                    assert not lab_vital_df['StudyID'].isna().any(), 'Some StudyIDs in lab_vital_df are not in patient_df'
                    # convert time to a consistent format
                    time_conversion_type = self._Labs_And_Vitals_Time_Conversion_Type_Dict[filename]
                    time_str_format = self._Labs_And_Vitals_Datetime_Format_Dict[filename]
                    match time_conversion_type if isinstance(time_conversion_type, str) else time_conversion_type[0]:
                        case 'datetime':
                            lab_vital_df['ObservationDtTm'] = pd.to_datetime(lab_vital_df['ObservationDtTm'], format=time_str_format)
                            if observation_smoother:
                                lab_vital_df = observation_smoother(lab_vital_df, key, filename)
                            lab_vital_df = self._Convert_Datetime_To_Hours_From_Admit(self._patient_df, 'ICU_AdmitDtTm', lab_vital_df, 'ObservationDtTm',
                                                                                      self._hours_preceding_icu_admit, self._hours_post_icu_admit)
                        case 'day_time':
                            lab_vital_df[time_conversion_type[1][1]] = pd.to_datetime(lab_vital_df[time_conversion_type[1][1]], format=time_str_format).dt.time
                            lab_vital_df = self._Create_ObsDt_From_Day_Time(self._patient_df, 'ICU_AdmitDtTm', lab_vital_df, time_str_format, *time_conversion_type[1], new_dt_col='ObservationDtTm')
                            if observation_smoother:
                                lab_vital_df = observation_smoother(lab_vital_df, key, filename)
                            lab_vital_df.drop(columns=['ObservationDtTm', 'ICU_AdmitDtTm'], inplace=True)
                            lab_vital_df = self._Convert_Day_Time_To_Hours_From_Admit(self._patient_df, 'ICU_AdmitDtTm', lab_vital_df, *time_conversion_type[1],
                                                                                      hours_before_icu_admit=self._hours_preceding_icu_admit, hours_after_icu_admit=self._hours_post_icu_admit)
                        case 'hours_from_admit':
                            if observation_smoother:
                                lab_vital_df = observation_smoother(lab_vital_df, key, filename)
                            lab_vital_df = self._Convert_Rename_Hours_From_Admit_Column(lab_vital_df, 'Hours_from_ICU_Admit', hours_preceding_icu_admit, hours_post_icu_admit)
                    # convert observation value type
                    match self._Labs_And_Vitals_Obs_Vals_Conversion_Dict[filename]:
                        case 'coerce':
                            lab_vital_df['Observation'] = pd.to_numeric(lab_vital_df['Observation'], errors='coerce')
                        case 'cast':
                            lab_vital_df['Observation'] = lab_vital_df['Observation'].astype(float)
                    # shift timesteps to start at 1
                    self._Shift_Timesteps(lab_vital_df, self._hours_preceding_icu_admit)
                    # update lab_vital_df_list and select columns of interest
                    lab_vital_df_list.append(lab_vital_df[['StudyID', 'Observation', 'timestep']])
                    assert not lab_vital_df['timestep'].isna().any(), f'Some timesteps in {filename} lab_vital_df are NaN'
                    assert not (lab_vital_df['timestep'] < 1).any(), f'Some timesteps in {filename} lab_vital_df are less than 1'
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    raise e
            # update labs_vitals dict
            self._labs_vitals_df_dict[key] = pd.concat(lab_vital_df_list, ignore_index=True)
        # load IVF data
        ivf_df_0 = pd.read_csv(os.path.join(base_path, 'MASTER_2012-2015_IVF_bolus_and_hourly.csv'))
        ivf_df_0 = ivf_df_0[ivf_df_0['StudyID'].isin(self._patient_df['StudyID'])]
        ivf_df_0.drop(columns=self._IVF_Drop_Columns['2012-2015'], inplace=True)
        ivf_df_0.rename(columns=self._IVF_Column_Rename_Dict['2012-2015'], inplace=True)
        ivf_df_0['ObservationDtTm'] = pd.to_datetime(ivf_df_0['ObservationDtTm'], format='%m/%d/%y %H:%M')
        ivf_df_1 = pd.read_csv(os.path.join(base_path, 'MASTER_2016-2019_IVF_all.csv'))
        ivf_df_1 = ivf_df_1[ivf_df_1['StudyID'].isin(self._patient_df['StudyID'])]
        ivf_df_1.drop(columns=self._IVF_Drop_Columns['2016-2019'], inplace=True)
        ivf_df_1.rename(columns=self._IVF_Column_Rename_Dict['2016-2019'], inplace=True)
        ivf_df_1['ObservationDtTm'] = pd.to_datetime(ivf_df_1['ObservationDtTm'], format='%m/%d/%y %I:%M %p')
        self._ivf_df = pd.concat([ivf_df_0, ivf_df_1], ignore_index=True)
        self._ivf_df = self._Convert_Datetime_To_Hours_From_Admit(self._patient_df, 'ICU_AdmitDtTm', self._ivf_df, 'ObservationDtTm',
                                                                  self._hours_preceding_icu_admit, None) #self._hours_preceding_icu_admit, self._hours_post_icu_admit)
        self._Shift_Timesteps(self._ivf_df, self._hours_preceding_icu_admit)
        assert not self._ivf_df['timestep'].isna().any(), 'Some timesteps in ivf_df are NaN'
        assert not (self._ivf_df['timestep'] < 1).any(), 'Some timesteps in ivf_df are less than 1'
        # load vasopressor data
        vasopressor_df_0 = pd.read_csv(os.path.join(base_path, 'MASTER_2012-2015_Pressors.csv'))
        vasopressor_df_0 = vasopressor_df_0[vasopressor_df_0['StudyID'].isin(self._patient_df['StudyID'])]
        vasopressor_df_0.drop(columns=['AdmitDate'], inplace=True)
        vasopressor_df_0.rename(columns={'EventEndDtTm': 'ObservationDtTm', 'Dose': 'Observation', 'Pressor': 'Vasopressor'}, inplace=True)
        vasopressor_df_0['ObservationDtTm'] = pd.to_datetime(vasopressor_df_0['ObservationDtTm'], format='%m/%d/%y %I:%M %p')
        vasopressor_df_1 = pd.read_csv(os.path.join(base_path, 'MASTER_2016-2019_Pressors.csv'))
        vasopressor_df_1 = vasopressor_df_1[vasopressor_df_1['StudyID'].isin(self._patient_df['StudyID'])]
        vasopressor_df_1.drop(columns=['AdmitDate'], inplace=True)
        vasopressor_df_1.rename(columns={'EventEndDtTm': 'ObservationDtTm', 'PressorDose': 'Observation', 'Pressor': 'Vasopressor'}, inplace=True)
        vasopressor_df_1['ObservationDtTm'] = pd.to_datetime(vasopressor_df_1['ObservationDtTm'], format='%m/%d/%y %H:%M')
        self._vasopressor_df = pd.concat([vasopressor_df_0, vasopressor_df_1], ignore_index=True)
        self._vasopressor_df = self._Convert_Datetime_To_Hours_From_Admit(self._patient_df, 'ICU_AdmitDtTm', self._vasopressor_df, 'ObservationDtTm',
                                                                          self._hours_preceding_icu_admit, None) #self._hours_preceding_icu_admit, self._hours_post_icu_admit)
        self._Shift_Timesteps(self._vasopressor_df, self._hours_preceding_icu_admit)
        assert not self._vasopressor_df['timestep'].isna().any(), 'Some timesteps in vasopressor_df are NaN'
        assert not (self._vasopressor_df['timestep'] < 1).any(), 'Some timesteps in vasopressor_df are less than 1'

    @staticmethod
    def _Convert_Datetime_To_Hours_From_Admit(patient_df: pd.DataFrame, patient_initial_time_col: str, data_df: pd.DataFrame, data_time_col: str, hours_before_icu_admit: Optional[int],
                                              hours_after_icu_admit: Optional[int], new_time_col: str = 'timestep') -> pd.DataFrame:
        """
        Converts a datetime column to an hours from admit column.
        """
        joined_df = data_df.merge(patient_df[['StudyID', patient_initial_time_col]], on='StudyID', how='left')
        assert not joined_df['StudyID'].isna().any(), 'Some StudyIDs in data_df are not in patient_df'
        joined_df[new_time_col] = np.ceil((joined_df[data_time_col] - joined_df[patient_initial_time_col]).dt.total_seconds() / 3600.0).astype(int)
        # filter timesteps
        if hours_before_icu_admit is None:
            before_mask = pd.Series(True, joined_df.index)
        else:
            before_mask = -hours_before_icu_admit < joined_df[new_time_col]  # hours before admit (0 is 1 hour before admit)
        if hours_after_icu_admit is None:
            after_mask = pd.Series(True, joined_df.index)
        else:
            after_mask = joined_df[new_time_col] <= hours_after_icu_admit
        joined_df = joined_df[before_mask & after_mask]
        return joined_df.drop(columns=[patient_initial_time_col, data_time_col])

    @staticmethod
    def _Create_ObsDt_From_Day_Time(patient_df: pd.DataFrame, patient_initial_time_col: str, data_df: pd.DataFrame, time_str_format: str, data_day_col: str, data_time_col: str, new_dt_col: str) -> pd.DataFrame:
        """
        Creates Observation Datatime from initial admit, day, and time columns.
        """
        joined_df = data_df.merge(patient_df[['StudyID', patient_initial_time_col]], on='StudyID', how='left')
        assert joined_df['StudyID'].isna().any() == 0, 'Some StudyIDs in data_df are not in patient_df'
        # convert day and time columns into a datetime column using patient_initial_time_col
        joined_df['combined_dt'] = joined_df[patient_initial_time_col] + joined_df[data_day_col].apply(lambda v: pd.Timedelta(days=v - 1))  # -1 because day 1 is the first day of ICU admission
        joined_df[new_dt_col] = joined_df[['combined_dt', data_time_col]].apply(lambda r: pd.Timestamp.combine(r['combined_dt'], r[data_time_col]), axis=1)
        return joined_df.drop(columns=['combined_dt'])

    @staticmethod
    def _Convert_Day_Time_To_Hours_From_Admit(patient_df: pd.DataFrame, patient_initial_time_col: str, data_df: pd.DataFrame, data_day_col: str, data_time_col: str, hours_before_icu_admit: Optional[int],
                                              hours_after_icu_admit: Optional[int], new_time_col: str = 'timestep') -> pd.DataFrame:
        """
        Converts day and time columns to a single hours from admit column.

        NOTES
        1. I believe day 1 is the first day of ICU admission and day 0 is the day before ICU admission.
        2. I believe the time is the time of day the measurement was recorded - NOT the time since ICU admission.
        """
        joined_df = data_df.merge(patient_df[['StudyID', patient_initial_time_col]], on='StudyID', how='left')
        assert joined_df['StudyID'].isna().any() == 0, 'Some StudyIDs in data_df are not in patient_df'
        # joined_df[data_time_col] = pd.to_datetime(joined_df[data_time_col], format='%I:%M:%S %p').dt.time
        # convert day and time columns into a datetime column using patient_initial_time_col
        joined_df['combined_dt'] = pd.to_datetime(joined_df[patient_initial_time_col].dt.date.astype(str) + ' ' + joined_df[data_time_col].astype(str)) + joined_df[data_day_col].apply(lambda v: pd.Timedelta(days=v-1)) # -1 because day 1 is the first day of ICU admission
        joined_df[new_time_col] = np.ceil((joined_df['combined_dt'] - joined_df[patient_initial_time_col]).dt.total_seconds() / 3600.0).astype(int)
        # joined_df[new_time_col] = np.ceil((joined_df[data_day_col] * 24.0 * 60 ** 2 + joined_df[data_time_col].apply(lambda t: (t.hour * 60 + t.minute) * 60 + t.second)) / 3600.0).astype(int)
        # filter timesteps
        if hours_before_icu_admit is None:
            before_mask = pd.Series(True, joined_df.index)
        else:
            before_mask = -hours_before_icu_admit < joined_df[new_time_col] # hours before admit (0 is 1 hour before admit)
        if hours_after_icu_admit is None:
            after_mask = pd.Series(True, joined_df.index)
        else:
            after_mask = joined_df[new_time_col] <= hours_after_icu_admit
        joined_df = joined_df[before_mask & after_mask]
        return joined_df.drop(columns=[patient_initial_time_col, data_day_col, data_time_col, 'combined_dt'])

    @staticmethod
    def _Convert_Rename_Hours_From_Admit_Column(data_df: pd.DataFrame, old_time_col: str, hours_before_icu_admit: int, hours_after_icu_admit: int, new_time_col: str = 'timestep') -> pd.DataFrame:
        """
        Renames an hours from admit column.
        """
        new_df = data_df.rename(columns={old_time_col: new_time_col})
        new_df[new_time_col] = new_df[new_time_col].astype(int)
        # filter timesteps
        before_mask = -hours_before_icu_admit < new_df[new_time_col] # hours before admit (0 is 1 hour before admit)
        after_mask = new_df[new_time_col] <= hours_after_icu_admit
        new_df = new_df[before_mask & after_mask]
        return new_df

    @staticmethod
    def _Shift_Timesteps(data_df: pd.DataFrame, shift_num: int, time_col: str = 'timestep'):
        """
        Shifts the timesteps of a dataframe by a specified amount.
        """
        data_df[time_col] += shift_num

    @property
    def patient_df(self) -> pd.DataFrame:
        return self._patient_df

    @property
    def labs_vitals_df_dict(self) -> Dict[str, pd.DataFrame]:
        return self._labs_vitals_df_dict

    @property
    def ivf_df(self) -> pd.DataFrame:
        # we dont do the before mask because this has already been accounted for and we shifted the timesteps
        # hours before admit (0 is 1 hour before admit)
        after_mask = self._ivf_df['timestep'] <= self._hours_post_icu_admit + self._hours_preceding_icu_admit # because we shifted
        ivf_view = self._ivf_df[after_mask]
        return ivf_view

    @property
    def raw_ivf_df(self) -> pd.DataFrame:
        return self._ivf_df

    @property
    def vasopressor_df(self) -> pd.DataFrame:
        # we dont do the before mask because this has already been accounted for and we shifted the timesteps
        # hours before admit (0 is 1 hour before admit)
        after_mask = self._vasopressor_df['timestep'] <= self._hours_post_icu_admit + self._hours_preceding_icu_admit # because we shifted
        vaso_view = self._vasopressor_df[after_mask]
        return vaso_view

    @property
    def raw_vasopressor_df(self) -> pd.DataFrame:
        return self._vasopressor_df

    @property
    def hours_preceding_icu_admit(self) -> int:
        return self._hours_preceding_icu_admit

    @property
    def hours_post_icu_admit(self) -> int:
        return self._hours_post_icu_admit

    @property
    def num_timesteps(self) -> int:
        return self._hours_preceding_icu_admit + self._hours_post_icu_admit

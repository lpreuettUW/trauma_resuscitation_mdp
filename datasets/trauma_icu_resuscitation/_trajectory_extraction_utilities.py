import re
import numpy as np
import pandas as pd
from typing import Set, Literal, Optional, Tuple, Dict


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter by
    - Require IVF_before_hr_24_no_L_cutoff >= 0.25 (i.e., 250 ml of IVF in first 24 hrs)
    - Exclude isolated head injuries (i.e., Head AIS >= 3 and all other AIS <= 1)
    :param df: compiled patient dataframe
    :return: filtered patient dataframe
    """
    max_ais_col_name = 'MaxNonHeadAIS'
    df[max_ais_col_name] = df[['MaxHeadAIS', 'MaxChestAIS', 'MaxAbdAIS', 'MaxSpineAIS', 'MaxLEAIS', 'MaxUEAIS', 'MaxFaceAIS', 'MaxNeckAIS']].max(
        axis=1)
    filtered_data = df[((df['IVF_before_hr_24_no_L_cutoff'] > 0.25) | np.isclose(df['IVF_before_hr_24_no_L_cutoff'], 0.25)) &
                       ((df['MaxHeadAIS'] < 3) | (df[max_ais_col_name] > 1))]
    df.drop(columns=[max_ais_col_name], inplace=True)
    return filtered_data


def load_patient_cohort_ids_with_icu_start_datetime(start_dt_col: str) -> pd.DataFrame:
    # load 2012-2015 patient cohort ids
    compiled_patient_data_2012_2015 = pd.read_csv(
        '<path_to_dataset>/2012-2015_compiled_patient_data_with_cohorts_no_L_required_new_IVF_EarlyLateBolus.csv')
    compiled_patient_data_2012_2015 = filter_data(compiled_patient_data_2012_2015)[['StudyID', start_dt_col]]
    # load 2016-2019 patient cohort ids
    compiled_patient_data_2016_2019 = pd.read_csv(
        '<path_to_dataset>/2016-2019_compiled_patient_data_with_cohorts_no_L_required_new_IVF_EarlyLateBolus.csv')
    compiled_patient_data_2016_2019['StudyID'] = compiled_patient_data_2016_2019['StudyID'].astype(int)
    compiled_patient_data_2016_2019 = filter_data(compiled_patient_data_2016_2019)[['StudyID', start_dt_col]]
    # combine patient cohort ids
    cohort_df_ = pd.concat([compiled_patient_data_2012_2015, compiled_patient_data_2016_2019])
    cohort_df_[start_dt_col] = pd.to_datetime(cohort_df_[start_dt_col], format='%Y-%m-%d %H:%M:%S')
    cohort_df_.rename(columns={start_dt_col: 'initial_date_time'}, inplace=True)
    print(
        f'Number of patient cohort ids: {cohort_df_.shape[0]} (2012-2015: {compiled_patient_data_2012_2015.shape[0]}, 2016-2019: {compiled_patient_data_2016_2019.shape[0]})')
    return cohort_df_


def get_dataset_file_path_pairs() -> Set[Tuple[Optional[str], Optional[str]]]:
    feature_file_pairs = {
        ('<path_to_dataset>/MASTER_2012-2015_Creatinine.csv',
         '<path_to_dataset>/MASTER_2016-2019_Creatinine.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_INR.csv',
         '<path_to_dataset>/MASTER_2016-2019_INR.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_pH.csv',
         '<path_to_dataset>/MASTER_2016-2019_pH.csv'),
        ('<path_to_dataset>/2012-2015_Corrected_FiO2.csv',
         '<path_to_dataset>/2016-2019_Corrected_FiO2.csv'),
        (None, '<path_to_dataset>/MASTER_2016-2019_RR.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_MAP.csv',
         '<path_to_dataset>/MASTER_2016-2019_MAP.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_HR.csv',
         '<path_to_dataset>/MASTER_2016-2019_HR.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_SBP.csv',
         '<path_to_dataset>/MASTER_2016-2019_SBP.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_DBP.csv',
         '<path_to_dataset>/MASTER_2016-2019_DBP.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_Temp.csv',
         '<path_to_dataset>/MASTER_2016-2019_Temp.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_Hourly_UOP_Total.csv',
         '<path_to_dataset>/MASTER_2016-2019_UOP_no_dupes.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_lactate.csv',
         '<path_to_dataset>/MASTER_2016-2019_lactate.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_PaO2.csv',
         '<path_to_dataset>/MASTER_2016-2019_PaO2.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_PaCO2.csv',
         '<path_to_dataset>/MASTER_2016-2019_PaCO2.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_SpO2.csv',
         '<path_to_dataset>/MASTER_2016-2019_SpO2.csv'),
        (None, '<path_to_dataset>/MASTER_2016-2019_HCO3.csv'),
        ('<path_to_dataset>/MASTER_2012-2015_IVF_bolus_and_hourly.csv',
         '<path_to_dataset>/MASTER_2016-2019_IVF_all.csv'),
    }
    return feature_file_pairs


def further_filter_patient_cohort_ids(patient_cohort_ids: Set[int],
                                      filter_type: Literal['require_all_features', 'require_all_but_vaso'] = 'require_all_features') -> Set[int]:
    """
    Select a subset of patient ids meeting additional criteria.

    Features considered:
    - Creatinine
    - INR
    - pH
    - Corrected FiO2
    - RR
    - MAP
    - HR
    - SBP
    - DBP
    - Temp
    - UOP
    - Lactate
    - PaO2
    - PaCO2
    - SpO2
    - HCO3
    - IVF
    - Vasopressors
    :param patient_cohort_ids: full patient cohort id set
    :param filter_type: additional criteria to apply
    :return: filtered subset of patient ids
    """
    filtered_patient_ids = patient_cohort_ids.copy()  # shallow copy is sufficient bc patient ids are of primitive type
    feature_file_pairs = get_dataset_file_path_pairs()
    if filter_type == 'require_all_features':
        feature_file_pairs.add(('<path_to_dataset>/MASTER_2012-2015_Pressors.csv',
                                '<path_to_dataset>/MASTER_2016-2019_Pressors.csv'))
    for feat_2012_2015_dfile_path, feat_2016_2019_dfile_path in feature_file_pairs:
        feat_2012_2015_df = None if feat_2012_2015_dfile_path is None else pd.read_csv(feat_2012_2015_dfile_path)
        feat_2016_2019_df = None if feat_2016_2019_dfile_path is None else pd.read_csv(feat_2016_2019_dfile_path)
        for df in (feat_2012_2015_df, feat_2016_2019_df):
            if df is not None:
                if 'STUDYID' in df.columns:
                    df.rename(columns={'STUDYID': 'StudyID'}, inplace=True)
                if df['StudyID'].isna().any():
                    df.dropna(subset=['StudyID'], inplace=True)
                if df['StudyID'].dtype != int:
                    df['StudyID'] = df['StudyID'].astype(int)
        try:
            if feat_2012_2015_df is not None and feat_2016_2019_df is not None:
                feat_data = pd.concat([feat_2012_2015_df['StudyID'].astype(int), feat_2016_2019_df['StudyID'].astype(int)])
            elif feat_2012_2015_df is not None:
                feat_data = feat_2012_2015_df['StudyID'].astype(int)
            else:
                feat_data = feat_2016_2019_df['StudyID'].astype(int)
            filtered_patient_ids = filtered_patient_ids.intersection(set(feat_data.tolist()))
        except KeyError as ke:
            print('Missing column in feature data file for', feat_2012_2015_dfile_path, feat_2016_2019_dfile_path)
            print(feat_2012_2015_df.columns)
            print(feat_2016_2019_df.columns)
            print(feat_data.columns)
            raise ke
        except TypeError as te:
            print('Error in concatenating feature data for', feat_2012_2015_dfile_path, feat_2016_2019_dfile_path)
            print(type(feat_data.values))
            raise te
    return filtered_patient_ids


def get_observation_date_time_col_name_dict() -> Dict[str, Set[str]]:
    observation_date_time_dict = {
        'ObservationDtTm': {'2012-2015_Corrected_FiO2', '2016-2019_Corrected_FiO2', '2012-2015_Creatinine', '2016-2019_Creatinine', '2012-2015_INR', '2016-2019_INR', '2012-2015_lactate', '2016-2019_lactate', '2012-2015_PaCO2', '2016-2019_PaCO2', '2012-2015_PaO2', '2016-2019_PaO2', '2012-2015_pH', '2016-2019_pH', '2016-2019_HCO3'},
        'EventEndDtTm': { '2012-2015_HR', '2016-2019_HR', '2012-2015_MAP', '2016-2019_MAP', '2012-2015_Pressors', '2016-2019_Pressors', '2012-2015_SBP', '2016-2019_SBP', '2012-2015_Temp', '2016-2019_Temp', '2016-2019_RR', '2016-2019_DBP', '2016-2019_UOP', '2012-2015_IVF', '2016-2019_IVF', },
        'EVENT_END_DT_TM': {'2016-2019_SpO2', }
    }
    return observation_date_time_dict

def get_observation_date_time_format_dict() -> Dict[str, Set[str]]:
    observation_date_time_format_dict = {
        '%m/%d/%y %I:%M %p': {'2016-2019_pH', '2012-2015_pH', '2012-2015_INR', '2016-2019_INR', '2016-2019_IVF', '2012-2015_lactate', '2016-2019_lactate', '2012-2015_PaCO2', '2016-2019_PaCO2', '2012-2015_PaO2', '2012-2015_Pressors', '2016-2019_HCO3', },
        '%m/%d/%y %H:%M': {'2012-2015_Creatinine', '2016-2019_Creatinine', '2012-2015_IVF', '2016-2019_PaO2', '2016-2019_Pressors', },
        '%Y-%m-%d %H:%M:%S': {'2016-2019_Corrected_FiO2', '2012-2015_HR', '2016-2019_HR', '2012-2015_MAP', '2016-2019_MAP', '2016-2019_RR', '2012-2015_SBP', '2016-2019_SBP', '2012-2015_Temp', '2016-2019_Temp', '2016-2019_DBP', '2016-2019_UOP', '2012-2015_Corrected_FiO2',}
    }
    return observation_date_time_format_dict

def get_column_rename_dict() -> Dict[str, Dict[str, str]]:
    column_rename_dict = {
        '2016-2019_HCO3': {'STUDYID': 'StudyID'}
    }
    return column_rename_dict

def extract_observation_date_time_col_name(filepath: str) -> str:
    if 'Corrected_FiO2' in filepath:
        return re.search(r'(201[26]-201[59])', filepath).group(1) + '_Corrected_FiO2'
    else:
        match = re.search(r'MASTER_(201[26]-201[59]_[^_]{2,10})[_\.]', filepath)
        assert match, f'No match found for observation date time column name for {filepath}'
        assert match.group(1) is not None, f'Match group 1 is None for {filepath} - match {match.group(0)}'
        return match.group(1)

def compute_trajectory_length_stats(patient_cohort_df: pd.DataFrame) -> Tuple[int, ...]:
    feature_file_pairs = get_dataset_file_path_pairs()
    observation_date_time_col_name_dict = get_observation_date_time_col_name_dict()
    observation_date_time_format_dict = get_observation_date_time_format_dict()
    column_rename_dict = get_column_rename_dict()
    # Exclude specified file paths
    files_missing_datetime = {
        '<path_to_dataset>/MASTER_2012-2015_DBP.csv',
        '<path_to_dataset>/MASTER_2012-2015_Hourly_UOP_Total.csv',
        '<path_to_dataset>/MASTER_2012-2015_SpO2.csv',
        '<path_to_dataset>/MASTER_2016-2019_SpO2.csv'
    }
    # TODO: handle excluded files
    max_hours_from_admit = 0
    min_hours_from_admit = 1000
    num_negs = num_pos = 0
    for filepath_pair in feature_file_pairs:
        for filepath in filepath_pair:
            if filepath is not None and filepath not in files_missing_datetime:
                feat_df = pd.read_csv(filepath)
                dt_col_key = extract_observation_date_time_col_name(filepath)
                dt_col = None
                for k, v in observation_date_time_col_name_dict.items():
                    if dt_col_key in v:
                        dt_col = k
                        break
                assert dt_col is not None, f'No matching observation date time column name found for {filepath} with key {dt_col_key}'
                dt_format_str = None
                for k, v in observation_date_time_format_dict.items():
                    if dt_col_key in v:
                        dt_format_str = k
                        break
                assert dt_format_str is not None, f'No matching observation date time format string found for {filepath} with key {dt_col_key}'
                try:
                    feat_df[dt_col] = pd.to_datetime(feat_df[dt_col], format=dt_format_str) # parse it
                    # feat_df[dt_col] = feat_df[dt_col].dt.strftime('%Y-%m-%d %H:%M:%S') # convert it
                except ValueError as ve:
                    print(f'Error in converting date time column {dt_col} in {filepath}')
                    print(feat_df[dt_col].head())
                    raise ve
                except KeyError as ke:
                    print(f'Error in converting date time column {dt_col} in {filepath}')
                    print(feat_df.columns)
                    raise ke
                if dt_col_key in column_rename_dict:
                    feat_df.rename(columns=column_rename_dict[dt_col_key], inplace=True)
                feat_df.dropna(subset=['StudyID'], inplace=True)
                try:
                    feat_df['StudyID'] = feat_df['StudyID'].astype(int)
                except KeyError as ke:
                    print(f'Error in converting StudyID to int in {filepath}')
                    raise ke
                feat_df = patient_cohort_df.merge(feat_df, on='StudyID', how='inner')
                feat_df['initial_date_time'] = pd.to_datetime(feat_df['initial_date_time'], format='%Y-%m-%d %H:%M:%S')
                try:
                    feat_df['Hours_from_ICU_Admit'] = np.ceil((feat_df[dt_col] - feat_df['initial_date_time']).dt.total_seconds() / 3600) # use ceil to match Dr. Beni's hour calculations in other files
                except TypeError as te:
                    print(f'Error in computing hours from ICU admit for {filepath}')
                    print(feat_df.head())
                    print(dt_col)
                    print(feat_df[dt_col].head())
                    print(feat_df['initial_date_time'].head())
                    raise te
                if (feat_df['Hours_from_ICU_Admit'] <= 0).any():
                    print(f'Negative hours from ICU admit for {filepath}')
                    num_negs += (feat_df['Hours_from_ICU_Admit'] <= 0).sum()
                    # print(feat_df.head())
                    # raise RuntimeError()
                num_pos += (feat_df['Hours_from_ICU_Admit'] > 0).sum()
                feat_df['Hours_from_ICU_Admit'] = feat_df['Hours_from_ICU_Admit'].astype(int)
                max_hours_from_admit = max(max_hours_from_admit, feat_df['Hours_from_ICU_Admit'].max())
                if feat_df.shape[0] > 0:
                    min_hours_from_admit = min(min_hours_from_admit, feat_df['Hours_from_ICU_Admit'].min())
    return max_hours_from_admit, min_hours_from_admit, num_negs, num_pos

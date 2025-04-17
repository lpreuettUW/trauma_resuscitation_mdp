import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Literal, Tuple, Set
from tqdm import tqdm
from pqdm.threads import pqdm
import matplotlib.pyplot as plt

from datasets.trauma_icu_resuscitation._data_manager import DataManager
from mdp.trauma_icu_resuscitation.state_spaces import discrete as DiscreteStateSpace
from mdp.trauma_icu_resuscitation.action_spaces import binary as BinaryActionSpace, discrete as DiscreteActionSpace


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter by
    - Require IVF_before_hr_24_no_L_cutoff >= 0.25 (i.e., 250 ml of IVF in first 24 hrs)
    - Exclude isolated head injuries (i.e., Head AIS >= 3 and all other AIS <= 1)

    Dr. O'Keefe suggests increasing the patient cutoff threshold. maybe 1 L.
    :param df: compiled patient dataframe
    :return: filtered patient dataframe
    """
    max_ais_col_name = 'MaxNonHeadAIS'
    df[max_ais_col_name] = df[['MaxHeadAIS', 'MaxChestAIS', 'MaxAbdAIS', 'MaxSpineAIS', 'MaxLEAIS', 'MaxUEAIS', 'MaxFaceAIS', 'MaxNeckAIS']].max(axis=1)
    # mask = (((df['IVF_before_hr_24_no_L_cutoff'] > 0.25) | np.isclose(df['IVF_before_hr_24_no_L_cutoff'], 0.25))
    #         & ((df['MaxHeadAIS'] < 3) | (df[max_ais_col_name] > 1))
    #         & (df['TraumaType'].str.strip().str.lower().isin({'b', 'p'})))
    # new criteria from Dr. O'Keefe
    mask = (df['InitLactate'] >= 2.0) | np.isclose(df['InitLactate'], 2.0)
    filtered_data = df[mask]
    df.drop(columns=[max_ais_col_name], inplace=True)
    return filtered_data


def correct_height(df: pd.DataFrame):
    """
    Correct height values
    :param df: compiled patient dataframe
    """
    # 300 cm is almost 10 feet - mark these patient heights as unknown
    df.loc[df['Height'] >= 300, 'Height'] = np.nan
    # everything below 100 cm is likely in inches - convert to cm
    df.loc[df['Height'] < 100, 'Height'] = df.loc[df['Height'] < 100, 'Height'].apply(lambda h: h * 2.54)
    return df


def compute_bmi(df: pd.DataFrame):
    """
    Compute BMI from height and weight
    :param df: compiled patient dataframe
    """
    df['BMI'] = 10000 * df['Weight'] / np.power(df['Height'], 2)
    return df


def correct_vasopressor_observations(df: pd.DataFrame):
    """
    Correct vasopressor observations
    :param df: compiled patient dataframe
    """
    norepinephrine_outlier_mask = df[(df['Vasopressor'] == 'norepinephrine') & (df['Observation'] >= 5000000)].index
    df.drop(norepinephrine_outlier_mask, inplace=True)
    # I think these outliers may have accidentally been recorded with an extra 0
    vasopressin_outlier_mask = df[(df['Vasopressor'] == 'vasopressin') & (df['Observation'] > 100)].index
    df.loc[vasopressin_outlier_mask, 'Observation'] = df.loc[vasopressin_outlier_mask, 'Observation'] / 10
    phenylephrine_outlier_mask = df[(df['Vasopressor'] == 'phenylephrine') & (df['Observation'] > 200)].index
    df.loc[phenylephrine_outlier_mask, 'Observation'] = df.loc[phenylephrine_outlier_mask, 'Observation'] / 10
    DOBUTamine_outlier_mask = df[(df['Vasopressor'] == 'DOBUTamine') & (df['Observation'] > 100)].index
    df.loc[DOBUTamine_outlier_mask, 'Observation'] = df.loc[DOBUTamine_outlier_mask, 'Observation'] / 10
    DOPamine_outlier_mask = df[(df['Vasopressor'] == 'DOPamine') & (df['Observation'] > 150)].index
    df.loc[DOPamine_outlier_mask, 'Observation'] = df.loc[DOPamine_outlier_mask, 'Observation'] / 10
    # EPINEPHrine: no changes
    # ePHEDrine: no changes
    # drop missing data
    df.drop(df[df['Observation'].isna()].index, inplace=True)
    return df


def correct_ivf_observations(df: pd.DataFrame):
    """
    Correct IVF observations

    Dr. Beni says the type of fluid doesnt matter - meaning we can directly combine the fluids so long as the units match.
    Units:
    - Dr. Beni says we should drop all mEq units
    - Dropping units in grams (g) in lieu of converting to ml
    - Coverting L to ml
    - Dropping units each, application, spray(s)
    Routes:
    - Dr. Beni says we should include routes in:
        - IV
        - IV Infusion
        - IVPB
        - IV Push
        - I.V. Push (non-std) Note: there are zero observations with this route
        - NaN
    :param df: compiled patient dataframe
    """
    # drop units each, application, spray(s), mEq, and g
    df.drop(df[df['Unit'].isin({'each', 'application', 'spray(s)', 'mEq', 'g'})].index, inplace=True)
    if False:
        # NOTE: Old logic to convert units to ml. Dr. Beni says we should drop all mEq units and I decided to remove grams for now.
        # convert mEq to ml. All fluids with mEq are sodium chloride. 4 mEq/mL according to https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=c44b0fdb-33df-4747-b443-5fb4deb456cb&type=display
        df.loc[df['Unit'] == 'mEq', 'Observation'] = df.loc[df['Unit'] == 'mEq', 'Observation'] / 4
        df.loc[df['Unit'] == 'mEq', 'Unit'] = 'mL'
        # convert grams of sodium chloride to ml. 2.17 g = 1 ml
        df.loc[(df['Unit'] == 'g') & (df['Type'] == 'sodium chloride'), 'Observation'] = df.loc[(df['Unit'] == 'g') & (df['Type'] == 'sodium chloride'), 'Observation'] / 2.17
        # convert grams of 70% dextrose to ml. Note: Im not really sure this one. I think it's 1 g = 1 ml. I need to confirm this.
        # No change. g = ml
        # convert grams of 20% dextrose to ml. Note: Im not really sure this one. I think it's 1 g = 1 ml. I need to confirm this.
        # No change. g = ml
        # update units for units in grams
        df.loc[df['Unit'] == 'g', 'Unit'] = 'mL'
    # convert liters to ml
    # 4000 L is likely a mistake. It's probably 4000 ml. So update everything less than 1000 L
    df.loc[(df['Unit'] == 'L') & (df['Observation'] < 1000), 'Observation'] = df.loc[(df['Unit'] == 'L') & (df['Observation'] < 1000), 'Observation'] * 1000
    df.loc[df['Unit'] == 'L', 'Unit'] = 'mL'
    assert (df['Unit'] == 'mL').all(), df['Unit'].unique()
    # drop routes that are not IV, IV Infusion, IVPB, IV Push, I.V. Push (non-std), or NaN
    df.drop(df[~df['Route'].isin({'IV', 'IV Infusion', 'IVPB', 'IV Push', 'I.V. Push (non-std)', np.nan})].index, inplace=True)
    # drop missing data
    df.drop(df[df['Observation'].isna()].index, inplace=True)


def correct_fio2_observations(df: pd.DataFrame):
    """
    Correct FiO2 observations
    :param df: compiled patient dataframe
    """
    mask = df['Observation'] > 100
    df.loc[mask, 'Observation'] = df.loc[mask, 'Observation'] / 100 # assume an extra zero was added to these values
    # Clean FiO2 for nasal cannula
    mask = df['Observation'] <= 9 | np.isclose(df['Observation'], 9)
    df.loc[mask, 'Observation'] = 100.0 * (df.loc[mask, 'Observation'] * 0.038 + 0.208)
    # Clean FiO2 for NRB
    mask = (df['Observation'] > 9) & (df['Observation'] <= 15) | np.isclose(df['Observation'], 15)
    df.loc[mask, 'Observation'] = df.loc[mask, 'Observation'] * 6

def cap_and_smooth_vital_df(df: pd.DataFrame, df_type: str, df_filename: str) -> pd.DataFrame:
    """
    Smooth vitals AND correct FiO2 observations
    :param df: lab or vital dataframe
    :param df_type: lab or vital name
    :param df_filename: filename of the dataframe
    """
    vitals_to_smooth_: Set[Literal['hr', 'map', 'sbp', 'dbp', 'rr', 'temp', 'fio2']] = {'hr', 'map', 'sbp', 'dbp', 'rr', 'temp', 'fio2'}
    vital_cap_dict = {
        'hr': 250,
        'map': 250,
        'sbp': 300,
        'dbp': 250,
        'fio2': 100,
    }
    if df_type in vitals_to_smooth_:
        match df_type:
            case 'fi02':
                correct_fio2_observations(df)
        # plot before
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        df['Observation'].plot.box(ax=ax[0])
        ax[0].set_title(f'{df_type} Original')
        ax[0].text(0.5, 0.5, f'Num NaNs: {df["Observation"].isna().sum()}', horizontalalignment='right', verticalalignment='bottom', transform=ax[0].transAxes)
        # apply cap
        cap = vital_cap_dict.get(df_type, None)
        if cap is not None:
            df.loc[df['Observation'] > cap, 'Observation'] = cap
        smooth_df = df.groupby(['StudyID', pd.Grouper(key='ObservationDtTm', freq='1h', origin='start')])[df.columns.tolist()].progress_apply(drop_vital_outliers).reset_index(drop=True)
        print(f'{df_type}: Num Observation NaNs in Original DF: {df["Observation"].isna().sum()} Smoothed DF: {smooth_df["Observation"].isna().sum()}')
        # plot after
        smooth_df['Observation'].plot.box(ax=ax[1])
        ax[1].set_title(f'{df_type} Smoothed')
        ax[1].text(0.5, 0.5, f'Num NaNs: {smooth_df["Observation"].isna().sum()}', horizontalalignment='right', verticalalignment='bottom', transform=ax[1].transAxes)
        fig.suptitle(f'{df_type} Boxplot Before and After Smoothing - # Obs: {smooth_df.shape[0]}')
        os.makedirs('<path_to_repo>/datasets/trauma_icu_resuscitation/preprocessed_cohort/smoothing_figures', exist_ok=True)
        fig.savefig(os.path.join('<path_to_repo>/datasets/trauma_icu_resuscitation/preprocessed_cohort/smoothing_figures', f'{df_filename}_boxplot_before_after_smoothing.png'))
        return smooth_df
    else:
        return df

def drop_lab_and_vital_missing_data(data_dict: Dict[str, pd.DataFrame]):
    """
    Drop missing data from labs and vitals
    :param data_dict: dictionary of labs and vitals dataframes
    """
    for key, df in data_dict.items():
        df.drop(df[df['Observation'].isna()].index, inplace=True)

def drop_vital_outliers(group, alpha=0.05, col='Observation'):
    """
    This function only gets values within the 5th and 95th percentiles (default), or other symmetric percentile range specified
    This is used in conjunction with grouping over a 2 hour time period to remove outliers in a fashion similar to what we do clinically
    NOTE: Dr. Beni's Code
    :param group:
    :param alpha:
    :param col:
    :return:
    """
    def trim_outliers_by_hour(x, upper_quantile, lower_quantile):
        if len(x) > 1:
            new_x = x.loc[(x[col] < upper_quantile) & (x[col] > lower_quantile)]
            if new_x.empty:
                new_x = x.copy()
                new_x[col] = x[col].mean()
            return new_x
        else:
            return x

    # upper_bound = group[col].quantile(1 - alpha)
    # lower_bound = group[col].quantile(alpha)
    q3 = group[col].quantile(0.75)
    q1 = group[col].quantile(0.25)
    iql = q3 - q1
    upper_bound = q3 + 1.5 * iql
    lower_bound = q1 - 1.5 * iql
    # if printout and (group['Hours_from_ICU_Admit'] >= 12).all() and (group['Hours_from_ICU_Admit'] < 20).all():
    # display(group)
    # display(group.groupby(['Hours_from_ICU_Admit']).apply(lambda x: trim_outliers_by_hour(x,upper_quantile,lower_quantile)).reset_index(drop=True))
    # return group.groupby(['timestep'])[group.columns.tolist()].apply(lambda x: trim_outliers_by_hour(x, upper_quantile, lower_quantile).reset_index(drop=True))
    # group['Observation'] = group['Observation'].clip(lower=lower_quantile, upper=upper_quantile) # this caps the values
    # this replaces the values with NaN
    group['Observation'] = group['Observation'].where(((group['Observation'] <= upper_bound) | np.isclose(group['Observation'], upper_bound))
                                                      & ((group['Observation'] >= lower_bound) | np.isclose(group['Observation'], lower_bound)), np.nan)
    return group


def fill_resuscitated_rewards_tensor(rewards: torch.FloatTensor, actions: torch.LongTensor, raw_ivf_view: pd.DataFrame, raw_vasopressor_view: pd.DataFrame, traj_start_ts: int, traj_end_ts: int, reward_scale: float = 100.0) -> int:
    """
    Fill resuscitated rewards tensor.

    :param rewards: rewards tensor
    :param actions: actions tensor
    :param raw_ivf_view: raw IVF view
    :param raw_vasopressor_view: raw vasopressor view
    :param died: whether the patient died
    :param traj_start_ts: trajectory start time step
    :param traj_end_ts: trajectory end time step
    :param reward_scale: reward scale
    :return: the truncated end time step
    """
    # fill resuscitated rewards tensor
    start_ts = raw_ivf_view['timestep'].min() - 1
    start_ts = min(start_ts, raw_vasopressor_view['timestep'].min() - 1)
    start_ts = min(start_ts, 1)
    end_ts = raw_ivf_view['timestep'].max()
    end_ts = max(end_ts, raw_vasopressor_view['timestep'].max())
    total_ts = 0 if np.isnan(start_ts) or np.isnan(end_ts) else (end_ts - start_ts + 1)
    # get all actions
    all_ivf_actions = torch.zeros(total_ts, dtype=torch.long)
    all_vasopressin_actions = torch.zeros(total_ts, dtype=torch.long)
    all_norepinephrine_actions = torch.zeros(total_ts, dtype=torch.long)
    if total_ts > 0:
        ivf_obs = raw_ivf_view.groupby('timestep').agg({'Observation': ['sum']})
        if ivf_obs.shape[0] > 0:
            all_ivf_actions[ivf_obs.index - 1] = torch.from_numpy(ivf_obs.map(lambda a: DiscreteActionSpace.IVF.Discretize(a).value).values[:, 0])
        if (raw_vasopressor_view['Vasopressor'] == 'vasopressin').any():
            vasopressin_obs = raw_vasopressor_view[raw_vasopressor_view['Vasopressor'] == 'vasopressin'].groupby('timestep').agg({'Observation': ['sum']})
            all_vasopressin_actions[vasopressin_obs.index - 1] = torch.from_numpy(vasopressin_obs.map(lambda a: DiscreteActionSpace.Vasopressin.Discretize(a).value).values[:, 0])
        if (raw_vasopressor_view['Vasopressor'] == 'norepinephrine').any():
            norepinephrine_obs = raw_vasopressor_view[raw_vasopressor_view['Vasopressor'] == 'norepinephrine'].groupby('timestep').agg({'Observation': ['sum']})
            all_norepinephrine_actions[norepinephrine_obs.index - 1] = torch.from_numpy(norepinephrine_obs.map(lambda a: DiscreteActionSpace.Norepinephrine.Discretize(a).value).values[:, 0])
        # now combine them
        all_any_action = all_ivf_actions.bool() | all_vasopressin_actions.bool() | all_norepinephrine_actions.bool()
        # now determine if the patient ever was resuscitated: look at hours between 25 and 96 to find last drug dose
        if all_any_action[24:96].any(): # 24 ICU admit start - we've already corrected for hour 1 being ICU start in the dataframe
            # if last dose was in [25, 72] then patient was resuscitated
            # if the last dose was after 72 then the patient was not resuscitated
            #last_dose_ts_in_48hr_window = all_any_action[24:72].nonzero().squeeze(-1)[-1]
            last_dose_ts_in_72hr_window = all_any_action[24:96].nonzero().squeeze(-1)[-1]
            if last_dose_ts_in_72hr_window + 1 + 24 < 72: # +1 to add one timestep of no dose to get last timestep, +24 to account for cutting the 24 hours prior to ICU admit out in line above
                # resuscitated
                end_ts = last_dose_ts_in_72hr_window + 1 + 24
                reward_sign = 1.0
            else:
                # not resuscitated
                end_ts = traj_end_ts
                reward_sign = -1.0
        else:
            # patient resuscitated before ICU - weird but lets give the reward at the second time step
            end_ts = traj_start_ts + 1
            reward_sign = 1.0
    else:
        # no actions taken
        end_ts = traj_start_ts + 1 # I guess they are resuscitated? weird
        reward_sign = 1.0
    # now fill the rewards tensor
    rewards[traj_start_ts:end_ts] = -1.0 # time penalty
    rewards[end_ts] = reward_sign * reward_scale
    return end_ts


def extract_patient_trajectory(patient_view_: pd.DataFrame, labs_and_vitals_views_dict_: Dict[str, pd.DataFrame], ivf_view_: pd.DataFrame, vasopressor_view_: pd.DataFrame, raw_ivf_view_: pd.DataFrame, raw_vasopressor_view_: pd.DataFrame,
                               action_space_str: Literal['binary', 'discrete'], reward_fn_str: Literal['sparse_mortality', 'sparse_vfd', 'resuscitated_w_time_penalty']) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]:
    """
    Extract patient trajectory
    :param patient_view_: patient view
    :param labs_and_vitals_views_dict_: dictionary of labs and vitals views
    :param ivf_view_: IVF view
    :param vasopressor_view_: vasopressor view
    :param raw_ivf_view_: raw IVF view - contains all IVF observations
    :param raw_vasopressor_view_: raw vasopressor view - contains all vasopressor observations
    :param action_space_str: string indicating binary or discrete action space
    :param reward_fn_str: string indicating reward function
    :return: states, actions
    """
    assert patient_view_.shape[0] == 1, 'Patient view should only have one row'
    # compute num timesteps
    n_timesteps = 72 # 24 hours before ICU admit + 48 hours in ICU - NOTE: some trajectories will have smaller trajectories
    # n_timesteps = ivf_view_['timestep'].max() if ivf_view_.shape[0] > 0 else 0
    # if vasopressor_view_.shape[0] > 0:
    #     n_timesteps = max(n_timesteps, vasopressor_view_['timestep'].max())
    # for key_, df_ in labs_and_vitals_views_dict_.items(): # num timesteps in labs and vitals views
    #     if df_.shape[0] > 0:
    #         n_timesteps = max(n_timesteps, df_['timestep'].max())
    # n_timesteps = int(n_timesteps)
    start_ts, end_ts = 999, 0
    # create actions tensor
    action_space = BinaryActionSpace if action_space_str == 'binary' else DiscreteActionSpace
    actions: torch.LongTensor = torch.zeros(n_timesteps, action_space.Components.n_components(), dtype=torch.long) # NOTE: zeros bc no drug values are 0
    try:
        # IVF
        ivf_obs = ivf_view_.groupby('timestep').agg({'Observation': ['sum']})
        # ivf_observations.index.astype(int).values
        if ivf_obs.shape[0] > 0:
            actions[ivf_obs.index - 1, action_space.Components.IVF.value] = torch.from_numpy(ivf_obs.map(lambda a: action_space.IVF.Discretize(a).value).values[:, 0])
        for action_comp_idx, action_comp, key in ((action_space.Components.Norepinephrine, action_space.Norepinephrine, 'norepinephrine'),
                                                  (action_space.Components.Vasopressin, action_space.Vasopressin, 'vasopressin')):
            mask = vasopressor_view_['Vasopressor'] == key
            if mask.any():
                obs = vasopressor_view_[mask].groupby('timestep').agg({'Observation': ['sum']})
                actions[obs.index - 1, action_comp_idx.value] = torch.from_numpy(obs.map(lambda a: action_comp.Discretize(a).value).values[:, 0])
        if ivf_view_.shape[0] > 0:
            start_ts = ivf_view_['timestep'].min() - 1
            end_ts = ivf_view_['timestep'].max() - 1
        elif vasopressor_view_.shape[0] > 0:
            start_ts = vasopressor_view_['timestep'].min() - 1
            end_ts = vasopressor_view_['timestep'].max() - 1
        # now truncate by Resuscitated criteria:
        #
    except Exception as e:
        raise RuntimeError(f'Error in extracting actions for patient {patient_view_["StudyID"].values[0]}: {e}')
    # create trajectory tensor
    states: torch.LongTensor = torch.zeros(n_timesteps, DiscreteStateSpace.Components.n_components(), dtype=torch.long)  # NOTE: zeros bc unknown values are 0
    try:
        for idx, comp_cls, key in ((DiscreteStateSpace.Components.Age, DiscreteStateSpace.Age, 'Age'),
                                    (DiscreteStateSpace.Components.Sex, DiscreteStateSpace.Sex, 'Male'),
                                    (DiscreteStateSpace.Components.Weight, DiscreteStateSpace.Weight, 'Weight'),
                                    (DiscreteStateSpace.Components.Height, DiscreteStateSpace.Height, 'Height'),
                                    (DiscreteStateSpace.Components.HeadAIS, DiscreteStateSpace.AIS, 'MaxHeadAIS'),
                                    (DiscreteStateSpace.Components.TraumaType, DiscreteStateSpace.TraumaType, 'TraumaType'),
                                    (DiscreteStateSpace.Components.InitialRBC, DiscreteStateSpace.InitialRBC, 'RBC_pre_ICU'),
                                    (DiscreteStateSpace.Components.InitialFFP, DiscreteStateSpace.InitialFFP, 'FFP_pre_ICU'),
                                    (DiscreteStateSpace.Components.InitialPlatelets, DiscreteStateSpace.InitialPlatelets, 'PLT_pre_ICU'),
                                    (DiscreteStateSpace.Components.InitialLactate, DiscreteStateSpace.Lactate, 'InitLactate'),
                                    (DiscreteStateSpace.Components.ICUAdmitLactate, DiscreteStateSpace.Lactate, 'lactate_value_ICU'),
                                    (DiscreteStateSpace.Components.VasopressorDoseWithinLastHour, DiscreteStateSpace.VasopressorDoseWithinLastHour, 'Pressors_ICU')):
            # fill static components for each timestep
            states[:, idx.value] = comp_cls.Discretize(patient_view_[key].values[0]).value

        # NOTE: timesteps are 1-indexed
        # fill labs and vitals
        lab_and_vital_discretization_tup = (
            (DiscreteStateSpace.Components.HR, DiscreteStateSpace.HR, 'hr', False, False, 'mean'),
            (DiscreteStateSpace.Components.SBP, DiscreteStateSpace.SBP, 'sbp', False, False, 'mean'),
            (DiscreteStateSpace.Components.DBP, DiscreteStateSpace.DBP, 'dbp', False, False, 'mean'),
            (DiscreteStateSpace.Components.MAP, DiscreteStateSpace.MAP, 'map', False, False, 'mean'),
            (DiscreteStateSpace.Components.SpO2, DiscreteStateSpace.SpO2, 'spo2', True, False, 'mean'),
            (DiscreteStateSpace.Components.UOP, DiscreteStateSpace.UOP, 'uop', True, False, 'sum'),
            (DiscreteStateSpace.Components.Temp, DiscreteStateSpace.Temp, 'temp', True, False, 'mean'),
            (DiscreteStateSpace.Components.HCO3, DiscreteStateSpace.HCO3, 'hco3', True, False, 'mean'),
            (DiscreteStateSpace.Components.Lactate, DiscreteStateSpace.Lactate, 'lactate', True, False, 'mean'),
            (DiscreteStateSpace.Components.Creatinine, DiscreteStateSpace.Creatinine, 'creatinine', True, True, 'mean'),
            (DiscreteStateSpace.Components.pH, DiscreteStateSpace.pH, 'ph', True, False, 'mean'),
            (DiscreteStateSpace.Components.INR, DiscreteStateSpace.INR, 'inr', True, False, 'mean'),
            (DiscreteStateSpace.Components.PaCO2, DiscreteStateSpace.PaCO2, 'paco2', True, False, 'mean'),
            (DiscreteStateSpace.Components.Hgb, DiscreteStateSpace.Hgb, 'hgb', True, True, 'mean'),
        )
        assert len(lab_and_vital_discretization_tup) == len(labs_and_vitals_views_dict_), 'Mismatch between lab and vital discretization tuple and labs and vitals views dictionary'
        for idx, comp_cls, key, need_convert, disc_need_male_param, agg_type in lab_and_vital_discretization_tup:
            df__ = labs_and_vitals_views_dict_[key]
            if df__.shape[0] > 0:
                # update start and stop timesteps
                start_ts = min(start_ts, df__['timestep'].min() - 1)
                end_ts = max(end_ts, df__['timestep'].max() - 1)
                # NOTE: timesteps are 1-indexed
                disc_kwargs = {'male': states[0, DiscreteStateSpace.Components.Sex.value].item()} if disc_need_male_param else dict()
                if need_convert:
                    apply_fn = lambda v: comp_cls.Discretize(v, **disc_kwargs).value
                else:
                    apply_fn = lambda v: int(comp_cls.Discretize(v, **disc_kwargs)) # ensure it's an integer - the dataset is wack son
                agg_df_view = df__.groupby('timestep').agg({'Observation': [agg_type]}) # combine observations for each timestep
                states[agg_df_view.index - 1, idx.value] = torch.from_numpy(agg_df_view.map(apply_fn).values[:, 0])
            # else: values are unknown - leave them at zero (zero is the unknown value)
    except Exception as e:
        raise RuntimeError(f'Error in extracting states for patient {patient_view_["StudyID"].values[0]}: {e}')
    if start_ts > end_ts:
        # support case where there's no data for the patient
        start_ts = 0
        end_ts = 1
    # cap start and end ts
    # start_ts = max(0, start_ts)
    # end_ts = min(n_timesteps - 1, end_ts)
    # create rewards tensor
    rewards: torch.FloatTensor = torch.zeros(n_timesteps, dtype=torch.float)
    match reward_fn_str:
        case 'sparse_mortality':
            # NOTE: rewards are zero for all timesteps except the last one
            rewards[end_ts] = -15.0 if patient_view_['Deceased'].values[0] else 15.0  # zero means they lived
        case 'sparse_vfd':
            # NOTE: rewards are zero for all timesteps except the last one
            reward = patient_view_['VentFreeDays'].values[0]
            if np.isclose(reward, 0.0): # zero means they died or were on the vent the whole time
                rewards[end_ts] = -25.0 # The mean VFD for surviving patients is 24.26284
            else:
                rewards[end_ts] = reward
        case 'resuscitated_w_time_penalty':
            end_ts = fill_resuscitated_rewards_tensor(rewards, actions, raw_ivf_view_, raw_vasopressor_view_, start_ts, end_ts)
            # update actions for new end_ts
            actions[end_ts+1:] = 0 # should be zero anyways...
        case _:
            raise ValueError(f'Unsupported reward function type: {reward_fn_str}')
    # create missing data tensor
    missing_data = torch.ones(n_timesteps, dtype=torch.bool)
    missing_data[start_ts:end_ts + 1] = False
    # create done data tensor
    dones = torch.zeros(n_timesteps, dtype=torch.bool)
    dones[end_ts] = True
    # correct for start ts
    if start_ts > 0:
        states = states.roll(-start_ts, dims=0)
        actions = actions.roll(-start_ts, dims=0)
        rewards = rewards.roll(-start_ts, dims=0)
        missing_data = missing_data.roll(-start_ts, dims=0)
        dones = dones.roll(-start_ts, dims=0)
    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0] == missing_data.shape[0] == n_timesteps, 'Mismatch in trajectory tensor shapes'
    assert states[~missing_data].shape[0] == actions[~missing_data].shape[0] == rewards[~missing_data].shape[0] == dones[~missing_data].shape[0], 'Mismatch in trajectory tensor shapes'
    # NOTE: missing data shouldnt be all zeros because the static components are applied to all timesteps
    assert actions[missing_data].isclose(torch.zeros(1, dtype=torch.long)).all(), 'Missing action data should be zero'
    assert rewards[missing_data].isclose(torch.zeros(1)).all(), 'Missing reward data should be zero'
    assert dones[missing_data].isclose(torch.zeros(1, dtype=torch.bool)).all(), 'Missing done data should be zero'
    assert missing_data.sum() == n_timesteps - (end_ts - start_ts + 1), 'Mismatch in missing data tensor'
    assert dones.sum() == 1, 'Only one done data point should be True'
    assert dones[end_ts - start_ts], 'Done data should be True for the last timestep'
    assert rewards[end_ts - start_ts] != 0, 'Reward should not be zero for the last timestep'
    match reward_fn_str:
        case 'sparse_mortality' | 'sparse_vfd':
            assert rewards.sum() == rewards[end_ts - start_ts], 'The only reward should be the last timestep reward'
        case 'resuscitated_w_time_penalty':
            assert rewards[:end_ts - start_ts].sum() == -1.0 * (end_ts - start_ts), 'The only reward should be the time penalty'
            assert rewards[end_ts - start_ts].abs().isclose(torch.tensor(100.0)), 'The last reward should have scale 100.0'
    assert rewards[dones].isclose(rewards[end_ts - start_ts]), 'Reward should be the same for the last timestep and the done timestep'
    if missing_data.any():
        assert (missing_data.nonzero().squeeze(-1) == torch.arange(end_ts - start_ts + 1, n_timesteps)).all(), 'Missing data should be contiguous'
    return states, actions, rewards, dones, missing_data


def extract_and_cache_patient_trajectory(arg_tup_: Tuple[int, DataManager, Literal['binary', 'discrete'], Literal['sparse_mortality', 'sparse_vfd', 'resuscitated_w_time_penalty'], str]):
    """
    Extract and cache patient trajectory
    :param arg_tup_: tuple of patient id, data manager, action space type, and reward function type - because pqdm is being stupid
    :param out_path_: output path
    """
    pid_, data_manager_, action_space_type_, reward_function_type_, out_path_ = arg_tup_
    # print(f'Extracting Trajectory for Patient {pid_}')
    # return pid_
    patient_view = data_manager_.patient_df[data_manager_.patient_df['StudyID'] == pid_]
    labs_and_vitals_views_dict = {key_: df_[df_['StudyID'] == pid_] for key_, df_ in data_manager_.labs_vitals_df_dict.items()}
    ivf_view = data_manager_.ivf_df[data_manager_.ivf_df['StudyID'] == pid_]
    raw_ivf_view = data_manager_.raw_ivf_df[data_manager_.raw_ivf_df['StudyID'] == pid_]
    vasopressor_view = data_manager_.vasopressor_df[data_manager_.vasopressor_df['StudyID'] == pid_]
    raw_vasopressor_view = data_manager_.raw_vasopressor_df[data_manager_.raw_vasopressor_df['StudyID'] == pid_]
    traj_states, traj_actions, traj_rewards, traj_dones, traj_missing = extract_patient_trajectory(patient_view, labs_and_vitals_views_dict, ivf_view, vasopressor_view,
                                                                                                   raw_ivf_view, raw_vasopressor_view,
                                                                                                   action_space_str=action_space_type_, reward_fn_str=reward_function_type_)
    traj_states, traj_actions, traj_rewards, traj_dones, traj_missing = traj_states.unsqueeze(0), traj_actions.unsqueeze(0), traj_rewards.unsqueeze(0), traj_dones.unsqueeze(0), traj_missing.unsqueeze(0)
    # save data
    torch.save(traj_states, f'{out_path_}/states_{pid_}.pt')
    torch.save(traj_actions, f'{out_path_}/{action_space_type_}_actions_{pid_}.pt')
    torch.save(traj_dones, f'{out_path_}/dones_{pid_}.pt')
    torch.save(traj_rewards, f'{out_path_}/{reward_function_type_}_rewards_{pid_}.pt')
    torch.save(traj_missing, f'{out_path_}/missing_{pid_}.pt')


if __name__ == '__main__':
    tqdm.pandas()
    base_path = '<path_to_dataset>'
    out_path = '<path_to_repo>/datasets/trauma_icu_resuscitation/preprocessed_cohort'
    os.makedirs(out_path, exist_ok=True)
    action_space_type: Literal['binary', 'discrete'] = 'discrete'
    reward_function_type: Literal['sparse_mortality', 'sparse_vfd', 'resuscitated_w_time_penalty'] = 'resuscitated_w_time_penalty'
    print('Extracting Trajectories with', action_space_type, 'action space and', reward_function_type, 'reward function')
    # this stuff is in cap_and_smooth_vital_df
    # vitals_to_smooth: Set[Literal['hr', 'map', 'sbp', 'dbp', 'rr', 'temp', 'fio2']] = {'hr', 'map', 'sbp', 'dbp', 'temp'} # 'rr', 'fio2'
    # if vitals_to_smooth:
    #     print('Smoothing vitals:', vitals_to_smooth)
    data_manager = DataManager(base_path, cohort_selector=filter_data, observation_smoother=cap_and_smooth_vital_df)
    # correct BMI
    correct_height(data_manager.patient_df)
    compute_bmi(data_manager.patient_df)
    # correct vasopressors
    correct_vasopressor_observations(data_manager.raw_vasopressor_df)
    # correct IVF
    correct_ivf_observations(data_manager.raw_ivf_df)
    print('IVF Fluid Types:', data_manager.ivf_df['Type'].unique())
    # drop lab and vital missing data
    drop_lab_and_vital_missing_data(data_manager.labs_vitals_df_dict)
    # print missing data
    print(f'Missing patient data {data_manager.patient_df.isna().sum()}')
    print(f'Missing vasopressor data {data_manager.vasopressor_df.isna().sum()}')
    print(f'Missing IVF data {data_manager.ivf_df.isna().sum()}')
    for key_, df_ in data_manager.labs_vitals_df_dict.items():
        print(f'Missing {key_} data {df_.isna().sum()}')

    pqdm_args = zip(data_manager.patient_df['StudyID'].unique(), [data_manager] * len(data_manager.patient_df['StudyID'].unique()),
                    [action_space_type] * len(data_manager.patient_df['StudyID'].unique()), [reward_function_type] * len(data_manager.patient_df['StudyID'].unique()),
                    [out_path] * len(data_manager.patient_df['StudyID'].unique()))
    parallelize = True
    if parallelize:
        result = pqdm(pqdm_args, extract_and_cache_patient_trajectory, n_jobs=4, desc='Extracting Trajectories', colour='blue', unit='patient')
    else:
        for arg_tup in tqdm(pqdm_args, desc='Extracting Trajectories', unit='patient', colour='blue'):
            extract_and_cache_patient_trajectory(arg_tup)
    # print(result)
    #error_results = [r for r in result if type(r) is not tuple]
    # if error_results:
    #     warnings.warn(f'Error in extracting trajectories for patients: {error_results}')
    # else:
    #     print('Trajectories Extracted Successfully')
    print('Trajectories Extracted Successfully')

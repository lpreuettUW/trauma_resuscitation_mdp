import os
import math
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from typing import Literal, Tuple, Optional, List, Final


def load_trauma_icu_resuscitation_data(split: int, load_type: Literal['iql', 'ope'], action_type: Literal['binary', 'discrete']) -> Tuple[Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor],
                                                                                                                                          Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor],
                                                                                                                                          Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor]]:
    """
    Load trauma ICU resuscitation data for a given split
    :param split: split number
    :param load_type: split type
    :param action_type: action space type
    :return: train, val, test data
    """
    # load split data
    split_base_path = '<path_to_dataset>/datasets/trauma_icu_resuscitation/stratified_splits'
    split_file = os.path.join(f'{split_base_path}', f'split_{split}.csv')
    split_data = pd.read_csv(split_file, index_col=0)
    longest_traj: Final[int] = 72 #int(split_data['traj_length'].max()) NOTE: traj_length is an old value - splits need to be updated...
    # load ivf data
    data_path = '<path_to_dataset>/datasets/trauma_icu_resuscitation/preprocessed_cohort'
    train_states, train_actions, train_rewards, train_dones, train_next_states, train_missing_data = list(), list(), list(), list(), list(), list()
    val_states, val_actions, val_rewards, val_dones, val_next_states, val_missing_data = list(), list(), list(), list(), list(), list()
    test_states, test_actions, test_rewards, test_dones, test_next_states, test_missing_data = list(), list(), list(), list(), list(), list()
    for split_states, split_actions, split_rewards, split_dones, split_next_states, split_missing_data, split_type in tqdm(zip([train_states, val_states, test_states],
                                                                                                                               [train_actions, val_actions, test_actions],
                                                                                                                               [train_rewards, val_rewards, test_rewards],
                                                                                                                               [train_dones, val_dones, test_dones],
                                                                                                                               [train_next_states, val_next_states, test_next_states],
                                                                                                                               [train_missing_data, val_missing_data, test_missing_data],
                                                                                                                               ['train', 'val', 'test']),
                                                                                                                           desc='Data Split', total=3, unit=' split', colour='green'):
        split_pids = split_data.loc[split_data['split'] == split_type, 'traj']
        for pid in split_pids:
            states = torch.load(os.path.join(data_path, f'states_{pid}.pt')).squeeze(0)
            actions = torch.load(os.path.join(data_path, f'{action_type}_actions_{pid}.pt')).squeeze(0)
            rewards = torch.load(os.path.join(data_path, f'resuscitated_w_time_penalty_rewards_{pid}.pt')).squeeze(0)
            dones = torch.load(os.path.join(data_path, f'dones_{pid}.pt')).squeeze(0)
            missing_data = torch.load(os.path.join(data_path, f'missing_{pid}.pt')).squeeze(0)
            next_states = states.roll(-1, dims=0)
            next_states[-1] = torch.zeros_like(next_states[-1]) # mark last next state as unknown - it should never be used, but just in case...
            # sanity check timesteps match
            assert states.size(0) == actions.size(0) == rewards.size(0) == dones.size(0) == next_states.size(0) == missing_data.size(0), f'Timestep mismatch: {states.size(0)}, {actions.size(0)}, {rewards.size(0)}, {dones.size(0)}, {next_states.size(0)}, {missing_data.size(0)}'
            if load_type == 'ope':
                # pad to match longest trajectory
                pad_len = longest_traj - states.size(0)
                states = torch.concat([states, torch.zeros(pad_len, states.size(1), dtype=torch.long)], dim=0).unsqueeze(0)
                actions = torch.concat([actions, torch.zeros(pad_len, actions.size(1), dtype=torch.long)], dim=0).unsqueeze(0)
                rewards = torch.concat([rewards, torch.zeros(pad_len, dtype=torch.float)], dim=0).unsqueeze(0)
                dones = torch.concat([dones, torch.zeros(pad_len, dtype=torch.bool)], dim=0).unsqueeze(0)
                next_states = torch.concat([next_states, torch.zeros(pad_len, next_states.size(1), dtype=torch.long)], dim=0).unsqueeze(0)
                missing_data = torch.concat([missing_data, torch.zeros(pad_len, dtype=torch.bool)], dim=0).unsqueeze(0)
                assert rewards[missing_data].sum().allclose(torch.zeros(1)), f'Missing data rewards should be zero: {rewards[missing_data]}'
            else:
                # mask out missing data for D3QN and IQL - they only use valid transitions
                states = states[~missing_data]
                actions = actions[~missing_data]
                rewards = rewards[~missing_data]
                dones = dones[~missing_data]
                next_states = next_states[~missing_data]
                missing_data = missing_data[~missing_data]
            # append to split data lists
            split_states.append(states)
            split_actions.append(actions)
            split_rewards.append(rewards)
            split_dones.append(dones)
            split_next_states.append(next_states)
            split_missing_data.append(missing_data)
    # concat tensors
    train_states = torch.cat(train_states, dim=0)
    train_actions = torch.cat(train_actions, dim=0)
    train_rewards = torch.cat(train_rewards, dim=0)
    train_dones = torch.cat(train_dones, dim=0)
    train_next_states = torch.cat(train_next_states, dim=0)
    train_missing_data = torch.cat(train_missing_data, dim=0)
    val_states = torch.cat(val_states, dim=0)
    val_actions = torch.cat(val_actions, dim=0)
    val_rewards = torch.cat(val_rewards, dim=0)
    val_dones = torch.cat(val_dones, dim=0)
    val_next_states = torch.cat(val_next_states, dim=0)
    val_missing_data = torch.cat(val_missing_data, dim=0)
    test_states = torch.cat(test_states, dim=0)
    test_actions = torch.cat(test_actions, dim=0)
    test_rewards = torch.cat(test_rewards, dim=0)
    test_dones = torch.cat(test_dones, dim=0)
    test_next_states = torch.cat(test_next_states, dim=0)
    test_missing_data = torch.cat(test_missing_data, dim=0)
    return ( # type: ignore
        (train_states, train_actions, train_rewards, train_dones, train_next_states, train_missing_data),
        (val_states, val_actions, val_rewards, val_dones, val_next_states, val_missing_data),
        (test_states, test_actions, test_rewards, test_dones, test_next_states, test_missing_data)
    )


def flatten_binary_actions(actions: torch.LongTensor) -> torch.LongTensor:
    """
    Flatten multidimensional actions tensor
    :param actions: multidimensional actions tensor
    :return: flattened actions tensor
    """
    def discretize_action(action: torch.LongTensor) -> int:
        if action.sum() == 0:
            return 0
        elif action[0] == 1 and action[1] == 0 and action[2] == 0:
            return 1
        elif action[0] == 0 and action[1] == 1 and action[2] == 0:
            return 2
        elif action[0] == 0 and action[1] == 0 and action[2] == 1:
            return 3
        elif action[0] == 1 and action[1] == 1 and action[2] == 0:
            return 4
        elif action[0] == 1 and action[1] == 0 and action[2] == 1:
            return 5
        elif action[0] == 0 and action[1] == 1 and action[2] == 1:
            return 6
        elif action[0] == 1 and action[1] == 1 and action[2] == 1:
            return 7
        else:
            raise ValueError(f'Invalid action: {action}')

    match actions.dim():
        case 2:
            new_actions = torch.zeros(actions.size(0), dtype=torch.long)
            for trans_idx in tqdm(range(actions.size(0)), desc='Flattening Transition Actions', total=actions.size(0), unit=' trans', colour='green', leave=False, position=0):
                cur_action = actions[trans_idx]
                new_actions[trans_idx] = discretize_action(cur_action)
        case 3:
            new_actions = torch.zeros(actions.size(0), actions.size(1), dtype=torch.long)
            for traj_idx in tqdm(range(actions.size(0)), desc='Flattening Trajectory Actions', total=actions.size(0), unit=' traj', colour='green', leave=False, position=0):
                for t_step in range(actions.size(1)):
                    cur_action = actions[traj_idx, t_step]
                    new_actions[traj_idx, t_step] = discretize_action(cur_action)
        case _:
            raise ValueError(f'Invalid actions tensor shape: {actions.shape}')
    return new_actions # type: ignore


def flatten_discrete_actions(actions: torch.LongTensor) -> torch.LongTensor:
    """
    Flatten multidimensional actions tensor for discrete action space
    :param actions: multidimensional actions tensor
    :return: flattened actions tensor
    """
    def discretize_action(action: torch.LongTensor) -> int:
        return action[0] * 4 + action[1] * 2 + action[2]

    match actions.dim():
        case 2:
            new_actions = torch.zeros(actions.size(0), dtype=torch.long)
            for trans_idx in tqdm(range(actions.size(0)), desc='Flattening Transition Actions', total=actions.size(0), unit=' trans', colour='green', leave=False, position=0):
                cur_action = actions[trans_idx]
                new_actions[trans_idx] = discretize_action(cur_action)
        case 3:
            new_actions = torch.zeros(actions.size(0), actions.size(1), dtype=torch.long)
            for traj_idx in tqdm(range(actions.size(0)), desc='Flattening Trajectory Actions', total=actions.size(0), unit=' traj', colour='green', leave=False, position=0):
                for t_step in range(actions.size(1)):
                    cur_action = actions[traj_idx, t_step]
                    new_actions[traj_idx, t_step] = discretize_action(cur_action)
        case _:
            raise ValueError(f'Invalid actions tensor shape: {actions.shape}')
    return new_actions

def unflatten_discrete_actions(actions: torch.LongTensor) -> torch.LongTensor:
    def discretize_action(action: torch.LongTensor) -> Tuple[int, int, int]:
        ivf_action = (action // 4).item()
        norepinephrine_action = (action.item() % 4) // 2
        vasopressin_action = action.item() % 2
        return ivf_action, norepinephrine_action, vasopressin_action

    match actions.dim():
        case 2:
            new_actions = torch.zeros(actions.size(0), 3, dtype=torch.long)
            for trans_idx in tqdm(range(actions.size(0)), desc='Flattening Transition Actions', total=actions.size(0), unit=' trans', colour='green', leave=False, position=0):
                cur_action = actions[trans_idx]
                new_act = discretize_action(cur_action)
                new_actions[trans_idx, 0] = new_act[0]
                new_actions[trans_idx, 1] = new_act[1]
                new_actions[trans_idx, 2] = new_act[2]
        case 3:
            new_actions = torch.zeros(actions.size(0), actions.size(1), 3, dtype=torch.long)
            for traj_idx in tqdm(range(actions.size(0)), desc='Flattening Trajectory Actions', total=actions.size(0), unit=' traj', colour='green', leave=False, position=0):
                for t_step in range(actions.size(1)):
                    cur_action = actions[traj_idx, t_step]
                    new_act = discretize_action(cur_action)
                    new_actions[traj_idx, t_step, 0] = new_act[0]
                    new_actions[traj_idx, t_step, 1] = new_act[1]
                    new_actions[traj_idx, t_step, 2] = new_act[2]
        case _:
            raise ValueError(f'Invalid actions tensor shape: {actions.shape}')
    return new_actions

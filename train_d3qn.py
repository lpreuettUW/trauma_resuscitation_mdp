import os
from nis import match

import torch
import mlflow
from tqdm import tqdm
from typing import Dict, Literal
from argparse import ArgumentParser

from agents.d3qn import D3QN
from mdp.trauma_icu_resuscitation.state_spaces.discrete import Components as StateSpaceComponents
from utilities.device_manager import DeviceManager
import utilities.trauma_icu_resuscitation_funcs
from utilities.implicit_qlearning_dataset import ImplicitQLearningDataset


def do_evaluation(agent: D3QN, dataloader_: torch.utils.data.DataLoader, num_samps: int, device_: torch.device, dataset_type_: str) -> Dict[str, float]:
    agent.eval()
    losses_dict = None
    for states, actions, next_states, rewards, dones in dataloader_: #tqdm(dataloader_, desc=f'{dataset_type_} Batch', total=len(dataloader_), unit='batch ', colour='blue'):
        states, actions, next_states, rewards, dones = states.to(device_), actions.to(device_), next_states.to(device_), rewards.to(device_), dones.to(device_)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        # agent will send data to device
        batch_losses_dict = agent.evaluate(states, actions, rewards, next_states, dones)
        if losses_dict is None:
            losses_dict = batch_losses_dict
        else:
            for key, value in batch_losses_dict.items():
                losses_dict[key] += value
    # log train losses
    for key in losses_dict.keys():
        losses_dict[key] /= num_samps # NOTE: we do this bc D3QN sums the losses
    return losses_dict


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    # agent
    policy_lr = 1e-4
    tau = 5e-3
    per_alpha = 0.6
    per_beta = 0.9
    per_eps = 1e-2
    hidden_size = 128
    reg_lambda = 5.0 # regularization lambda penalizing Q-values greater than the max possible reward
    # env
    gamma = 1.0
    num_train_steps = 60000 # Raghu et al. (2017) used 60k train steps
    action_type: Literal['binary', 'discrete'] = 'discrete'
    num_actions = 12 if action_type == 'discrete' else 8
    # training
    batch_size = 32
    val_step_mod = 1000
    log_mod = 100
    num_splits = 10
    initial_weights = None

    # mlflow stuffs
    mlflow_path = os.path.join('file:///', '<path_to_mlruns>', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = f'D3QN Trauma ICU: {action_type} Actions'
    run_name = 'Reward Function: sparse - scale = 15.0'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'policy_lr': policy_lr,
            'per_alpha': per_alpha,
            'per_beta': per_beta,
            'per_eps': per_eps,
            'hidden_size': hidden_size,
            'tau': tau,
            'gamma': gamma,
            'num_train_steps': num_train_steps,
            'reward_fn_name': 'sparse',
            'batch_size': batch_size,
            'val_step_mod': val_step_mod,
            'log_mod': log_mod,
            'num_splits': num_splits,
        }
        mlflow.log_params(param_dict)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split ', colour='green'): #, position=0, leave=True):
            # load data
            (
                (train_states, train_actions, train_rewards, train_dones, train_next_states, _),
                (val_states, val_actions, val_rewards, val_dones, val_next_states, _),
                (_, _, _, _, _, _)
            ) = utilities.trauma_icu_resuscitation_funcs.load_trauma_icu_resuscitation_data(split, 'iql', action_type)
            # flatten actions for D3QN
            match action_type:
                case 'discrete':
                    train_actions = utilities.trauma_icu_resuscitation_funcs.flatten_discrete_actions(train_actions)
                    val_actions = utilities.trauma_icu_resuscitation_funcs.flatten_discrete_actions(val_actions)
                case 'binary':
                    train_actions = utilities.trauma_icu_resuscitation_funcs.flatten_binary_actions(train_actions)
                    val_actions = utilities.trauma_icu_resuscitation_funcs.flatten_binary_actions(val_actions)
                case _:
                    raise ValueError(f'Invalid action type: {action_type}')
            # create dataloaders
            train_dataset = ImplicitQLearningDataset(train_states.float(), train_actions, train_next_states.float(), train_rewards, train_dones)
            val_dataset = ImplicitQLearningDataset(val_states.float(), val_actions, val_next_states.float(), val_rewards, val_dones)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # create agent
            d3qn_agent = D3QN(state_dim=len(StateSpaceComponents), action_dim=1, hidden_dim=hidden_size, reward_max=1.0,
                              gamma=gamma, tau=tau, lr=policy_lr, buffer_size=len(train_dataset), per_alpha=per_alpha,
                              per_beta=per_beta, per_eps=per_eps, reg_lambda=reg_lambda, batch_size=batch_size, num_actions=num_actions) # NOTE: num_actions isn't used in D3QN...
            d3qn_agent.reset_prioritization_bias_correction_annealing(num_train_steps)
            # load/cache initial weights
            if initial_weights is None:
                initial_weights = d3qn_agent.get_weights()
            else:
                d3qn_agent.load_weights(initial_weights)
            # fill replay buffer
            d3qn_agent.fill_replay_buffer(train_dataloader)
            # train agent
            best_weights = best_combined_loss = None  # TODO: I dont trust the loss to help us choose this
            for batch in range(num_train_steps): # tqdm(range(num_train_steps), desc='Batch', total=num_train_steps, unit=' batch', colour='blue'):
                batch_losses_dict = d3qn_agent.batch_train()
                if (batch + 1) % log_mod == 0:
                    # update loss dict keys
                    batch_losses_dict = {f'train_{key}_split_{split}': value for key, value in batch_losses_dict.items()}
                    mlflow.log_metrics(batch_losses_dict, step=batch)
            # evaluate agent
            for dataloader, num_samples, dataset_type in zip([train_dataloader, val_dataloader], [len(train_dataset), len(val_dataset)], ['train', 'val']):
                loss_dict = do_evaluation(d3qn_agent, dataloader, len(train_dataset), device, dataset_type)
                loss_dict = {f'eval_{dataset_type}_{key}_split_{split}': value for key, value in loss_dict.items()}
                mlflow.log_metrics(loss_dict, step=batch)
            # save model
            d3qn_agent.save_model(f'final_d3qn_model_split_{split}')


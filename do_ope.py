import os
import torch
import mlflow
from tqdm.auto import tqdm
from argparse import ArgumentParser

from ope.magic import MAGIC
from ope.fqe import FittedQEvaluation
from ope.behavior_policy_value import BehaviorPolicyValue
from mdp.trauma_icu_resuscitation.state_spaces.discrete import Components as StateSpaceComponents
from agents.d3qn import D3QN
from agents.implicit_q_learning import ImplicitQLearning
from agents.no_action_agent import NoActionAgent
from agents.random_action_agent import RandomActionAgent
from utilities.device_manager import DeviceManager
from utilities import trauma_icu_resuscitation_funcs
from utilities.ope_trajectory_dataset import OPETrajectoryDataset


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    ap.add_argument('--run_type', type=str, choices=['fqe', 'magic', 'both'], help='specify the type of run to perform (fqe, magic, both)')
    ap.add_argument('--fqe_run_id', type=str, help='specify the run ID of the FQE model to use')
    ap.add_argument('--action_type', type=str, choices=['binary', 'discrete'], help='specify the type of action space to use (binary, discrete)')
    ap.add_argument('--agent_type', type=str, choices=['d3qn', 'iql', 'tt', 'no_action', 'random'], help='specify the type of agent to use (d3qn, iql, tt, no_action, random)')
    ap.add_argument('--agent_runid', type=str, help='specify the run ID of the agent to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    num_splits = 10
    gamma = 1.0
    # fqe
    fqe_k_itrs = 100
    fqe_convergence_eps = 1e-3
    fqe_max_train_itrs = 25
    fqe_lr = 1e-4
    fqe_hidden_size = 128
    fqe_batch_size = 8192
    # magic
    magic_j_steps = {float('inf'), -1, 24, 48}
    magic_k_conf_iters = 2000
    magic_batch_size = 2 # extremely small batch size to avoid memory issues
    magic_eps = 1e-6
    # iql settings
    iql_policy_lr = 1e-4
    iql_critic_lr = 1e-4
    iql_expectile_val_lr = 1e-4
    iql_policy_hidden_size = 128
    iql_critic_hidden_size = 128
    iql_expectile_hidden_size = 128
    iql_agent_weight_decay = 1e-4
    iql_expectile = 0.8
    iql_temperature = 0.1
    iql_clip_norm = 1.0
    iql_tau = 5e-3
    # d3qn
    d3qn_policy_lr = 1e-4
    d3qn_tau = 5e-3
    d3qn_per_alpha = 0.6
    d3qn_per_beta = 0.9
    d3qn_per_eps = 1e-2
    d3qn_reg_lambda = 5.0
    d3qn_hidden_size = 128

    num_actions = 12 if args.action_type == 'discrete' else 8

    mlflow_path = os.path.join('file:///', '<path_to_mlruns>', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = ('FQE Retrospective' if args.run_type == 'fqe' else 'MAGIC Retrospective') + f'{args.action_type} Actions'
    run_name = f'{args.agent_type} - Reward Function: sparse'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'fqe_k_itrs': fqe_k_itrs,
            'fqe_convergence_eps': fqe_convergence_eps,
            'fqe_max_train_itrs': fqe_max_train_itrs,
            'fqe_lr': fqe_lr,
            'fqe_hidden_size': fqe_hidden_size,
            'fqe_batch_size': fqe_batch_size,
            'gamma': gamma,
            'reward_fn_name': 'sparse',
            'num_splits': num_splits,
            'agent_type': args.agent_type,
            'agent_runid': args.agent_runid,
            'fqe_run_id': args.fqe_run_id,
            'magic_j_steps': magic_j_steps,
            'magic_k_conf_iters': magic_k_conf_iters,
            'magic_batch_size': magic_batch_size,
            'magic_eps': magic_eps,
        }
        mlflow.log_params(param_dict)
        control_variates = None
        splits = range(num_splits) # [5]
        for split in tqdm(splits, desc='Split', total=num_splits, unit='split ', colour='green', position=0, leave=True):
            # load data
            (
                (_, _, _, _, _, _),
                (_, _, _, _, _, _),
                (test_states, test_actions, test_rewards, test_dones, test_next_states, test_missing_data_mask)
            ) = trauma_icu_resuscitation_funcs.load_trauma_icu_resuscitation_data(split, 'ope', args.action_type)
            # flatten actions for D3QN
            match args.action_type:
                case 'discrete':
                    test_actions = trauma_icu_resuscitation_funcs.flatten_discrete_actions(test_actions)
                case 'binary':
                    test_actions = trauma_icu_resuscitation_funcs.flatten_binary_actions(test_actions)
                case _:
                    raise ValueError(f'Invalid action type: {args.action_type}')
            # create trajectory dataset
            test_traj_dataset = OPETrajectoryDataset(test_states.float(), test_states.float(), test_actions, test_actions.float().unsqueeze(-1), test_next_states.float(),
                                                     test_next_states.float(), test_rewards, test_dones, test_missing_data_mask, num_actions=num_actions, flatten=True)
            match args.agent_type:
                case 'iql':
                    # load IQL agent
                    agent = ImplicitQLearning(state_dim=len(StateSpaceComponents), action_dim=1,
                                              num_actions=num_actions, policy_hidden_dim=iql_policy_hidden_size,
                                              critic_hidden_dim=iql_critic_hidden_size, expectile_val_hidden_dim=iql_expectile_hidden_size, policy_lr=iql_policy_lr,
                                              critic_lr=iql_critic_lr, expectile_val_lr=iql_expectile_val_lr,
                                              gamma=gamma, expectile=iql_expectile, temperature=iql_temperature, clip_norm=iql_clip_norm, tau=iql_tau)
                    agent.load_model(f'runs:/{args.agent_runid}/final_iql_model_split_{split}')
                    agent.eval()
                case 'd3qn':
                    agent = D3QN(state_dim=len(StateSpaceComponents), action_dim=1, hidden_dim=d3qn_hidden_size, reward_max=1.0,
                                 gamma=gamma, tau=iql_tau, lr=iql_policy_lr, buffer_size=0, per_alpha=d3qn_per_alpha,
                                 per_beta=d3qn_per_beta, per_eps=d3qn_per_eps, reg_lambda=d3qn_reg_lambda, batch_size=0,
                                 num_actions=num_actions)
                    agent.load_model(f'runs:/{args.agent_runid}/final_d3qn_model_split_{split}')
                    agent.eval()
                case 'tt':
                    raise NotImplementedError('TT not implemented')
                case 'no_action':
                    agent = NoActionAgent(no_action_val=0, num_actions=num_actions)
                case 'random':
                    agent = RandomActionAgent(no_action_val=0, num_actions=num_actions, available_actions=test_actions.unique())
                case _:
                    raise ValueError(f'Invalid agent type: {args.agent_type}')
            behavior_policy_value = BehaviorPolicyValue.ComputeValue(test_traj_dataset, batch_size=magic_batch_size, gamma=gamma) # TODO: add behavior_policy_value_batch_size
            print(f'({split}) Behavior Policy Value: {behavior_policy_value:.5f}')
            mlflow.log_metric('behavior_policy_value', behavior_policy_value, step=split)
            # create FQE
            fqe = FittedQEvaluation(test_traj_dataset, agent, k_itrs=fqe_k_itrs, convergence_eps=fqe_convergence_eps, max_train_itrs=fqe_max_train_itrs,
                                    lr=fqe_lr, hidden_size=fqe_hidden_size, batch_size=fqe_batch_size, use_behavior_policy_states=False,
                                    run_id=args.fqe_run_id, split_num=split)
            if not args.fqe_run_id:
                fqe.log_model()
            policy_value = fqe.compute_value()
            print(f'({split}) FQE Policy Value: {policy_value:.5f}')
            mlflow.log_metric('fqe', policy_value, step=split)
            match args.run_type:
                case 'magic' | 'both':
                    # create MAGIC
                    magic = MAGIC(test_traj_dataset, agent, gamma, magic_batch_size, fqe, magic_j_steps, magic_k_conf_iters, magic_eps)
                    policy_value, cur_control_variates = magic.compute_value()
                    if control_variates is None:
                        control_variates = cur_control_variates
                    else:
                        control_variates = torch.concat((control_variates, cur_control_variates), dim=0)
                    print(f'({split}) MAGIC Policy Value: {policy_value:.5f}')
                    print(f'({split}) J-Steps: {magic.j_steps}')
                    print(f'({split}) J-Step Weights: {magic.j_step_weights}')
                    mlflow.log_metric('magic', policy_value, step=split)
                    mlflow.log_metrics({f'j_step_{j_step}': weight for j_step, weight in zip(magic.j_steps, magic.j_step_weights)}, step=split)
        match args.run_type:
            case 'magic' | 'both':
                # compute control variate stats
                control_variate_means = control_variates.mean(dim=0)
                control_variate_stds = control_variates.std(dim=0)
                for j_step, mean, std in zip(magic.j_steps, control_variate_means, control_variate_stds):
                    print(f'Control Variate (J-Step: {j_step}): {mean:.5f} +/- {std:.5f}')
                    mlflow.log_metric(f'control_variate_mean_{j_step}', mean)
                    mlflow.log_metric(f'control_variate_std_{j_step}', std)

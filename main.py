"""
This file contains the main entry point of the project where parameters can be set.
"""


import datetime
import time
import os
import psutil
import torch
from torch import optim
# from torch.utils.tensorboard import SummaryWriter

# from ppo.network import ActorCriticNetwork
from ppo.network import ActorNetwork
from ppo.network import CriticNetwork
from ppo.environment import WindTunnelEnv
from ppo.training import PPOTraining

from utils import mkdir
from utils import create_file
from utils import write_file

# from tests.test_environments.test_env_01 import TestEnv01  # environment to test ppo implementation
# from tests.test_environments.test_env_02 import TestEnv02  # environment to test ppo implementation
from tests.test_environments.test_env_03 import TestEnv03  # environment to test ppo implementation
# from tests.test_environments.test_env_04 import TestEnv04  # environment to test ppo implementation

### Parameter Configuration ###

HIDDEN_SIZE         = 64        # Number of neurons in hidden layer in NN
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-5      # Learning rate for adam optimizer
GAMMA               = 0.75      # Discount factor to calculate returns
GAE_LAMBDA          = 0.95     # Smoothing factor for GAE (high = more accuracy and dependence on future rewards; lower = less variance but higher bias and dependence on immediate rewards)
EPSILON_CLIP        = 0.5       # Used to clip the ratio between new and old policy
ENTROPY_BETA        = 0.000      # Amount of importance given to the entropy bonus (low = stronger exploitation; high = stronger exploration)
PPO_STEPS           = 1024       # Number of transitions sampled for each training iteration = batch_size (should be multiple of MINI_BATCH_SIZE)
PPO_UPDATE_EPOCHS   = 100         # Number of pass over entire batch of training data
PPO_TRAIN_EPOCHS    = 20000    # Limit epochs for the PPO training loop
SAVE_INTERVAL       = 50        # interval for model checkpoints

LOAD_MODEL          = True     # Select to load an existing torch model

PARAMS_FILE         = "parameter.txt"
BATCH_FILE          = "batch_data.txt"
UPDATE_FILE         = "update_data.txt"
TEST_FILE           = "model_test.txt"

MODEL_LOAD_PATH_ACTOR   = "./logs/run_2025_03_25_16_09_19/model_checkpoints/epoch_200_torch_actor_model"
MODEL_LOAD_PATH_CRITIC  = "./logs/run_2025_03_25_16_09_19/model_checkpoints/epoch_200_torch_critic_model"

if __name__ == "__main__":
    
    # ### Increas codes priority (admin/root privileges needed) ###
    pid = os.getpid()  # get process ID
    print(f'\nProcess ID: {pid}\n')

    process = psutil.Process(pid)  # get current process

    # use this for high priority
    if os.name == 'nt':  # Windows
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Increased priority of Process")
    else:  # Linux/Mac (needs root privilege)
        process.nice(-10)

    # # use this for even higher priority (caution this can starve other processes !!!)
    # if os.name == 'nt':  # Windows
    #     process.nice(psutil.REALTIME_PRIORITY_CLASS)
    #     print("Increased priority of Process")
    # else:  # Linux/Mac (needs root privilege)
    #     process.nice(-20)

    ### NI USB-6281 I/O channel names ###
    names_output_channel = ["Dev1/ao0", "Dev1/ao1"]  # ao0: reservoir; ao1: valves
    num_actions = 1

    # names_input_channel = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
    #                         "Dev1/ai4", "Dev1/ai5", "Dev1/ai6",  "Dev1/ai7", "Dev1/ai8"]  # ai0: volume flow; ai1 - ai8: tau
    # num_states = 8

    # names_input_channel = ["Dev1/ai0", "Dev1/ai5", "Dev1/ai6",  "Dev1/ai7"]  # ai0: volume flow; ai1 - ai8: tau
    # num_states = 3

    names_input_channel = ["Dev1/ai0", "Dev1/ai7"]  # ai0: volume flow; ai1 - ai8: tau
    num_states = 1

    ### Create log folder ###
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    models_dir = mkdir('.', f'logs/run_{date}/model_checkpoints')
    log_dir = mkdir('.', f'logs/run_{date}')
    
    # Create log file for used Parameters
    fname_params = f'{log_dir}/{PARAMS_FILE}'
    create_file(fname_params, ['HIDDEN_SIZE', 'ACTOR_LEARNING_RATE', 'CRITIC_LEARNING_RATE', 'GAMMA',
                               'GAE_LAMBDA', 'PPO_EPSILON_CLIP', 
                               'ENTROPY_BETA', 'PPO_STEPS', 
                               'PPO_UPDATE_EPOCHS', 'PPO_TRAIN_EPOCHS'])
    
    # Create log file for PPO Batch Data
    fname_batch = f'{log_dir}/{BATCH_FILE}'
    action_names = ["Action"]
    state_names = names_input_channel[1:]  # List with names of mems sensor channels
    tau_state_names = [f'tau_{sn}' for sn in state_names]
    gamma_state_names = [ f'epoch_gamma_{sn}' for sn in state_names]
    create_file(fname_batch, ["frame_idx", "train_epoch", "step", 
                              "epoch_total_reward", "gae_state_return", 
                              "mc_state_return", "own_state_return", "state_value", "reward",
                              "normalized_gae_advantage", 
                              "normalized_mc_advantage", "normalized_own_advantage", "act_log_prob",
                              "dist_prob"] + action_names + ["count_valve_open"] + state_names + ["volume_flow"] + tau_state_names +
                ["epoch_mean_gamma"] + gamma_state_names)
    
    # Create log file for PPO Update Data
    fname_update = f'{log_dir}/{UPDATE_FILE}'
    create_file(fname_update, ["frame_idx", "train_epoch", "update_epoch", "actor_loss", "critic_loss", "entropy_loss", "mean_entropy"])
    
    ### Write used Parameters to file ###
    write_file(f'{log_dir}/{PARAMS_FILE}', 
               [f'{HIDDEN_SIZE}', f'{ACTOR_LEARNING_RATE}', f'{CRITIC_LEARNING_RATE}', f'{GAMMA}',
                f'{GAE_LAMBDA}', f'{EPSILON_CLIP}', 
                f'{ENTROPY_BETA}', f'{PPO_STEPS}', 
                f'{PPO_UPDATE_EPOCHS}', f'{PPO_TRAIN_EPOCHS}'])
    
    ### Autodetect CUDA ###
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\nDevice: {device}\n')
    
    ### Prepare environment ###
    env = WindTunnelEnv(num_states=num_states, num_actions=num_actions,
                        output_channel=names_output_channel,
                        input_channel=names_input_channel)
    
    # env = TestEnv03(num_states=num_states, num_actions=num_actions,
    #                 output_channel=names_output_channel,
    #                 input_channel=names_input_channel)  # TEST

    ### Prepare Networks ###
    actor_model = ActorNetwork(num_inputs=num_states, num_outputs=num_actions, hidden_size=HIDDEN_SIZE).to(device)
    print(f'Actor Model:\n\n{actor_model}\n')
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=ACTOR_LEARNING_RATE)
    
    critic_model = CriticNetwork(num_inputs=num_states, hidden_size=HIDDEN_SIZE).to(device)
    print(f'Critic Model:\n\n{critic_model}\n')
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=CRITIC_LEARNING_RATE)
    
    ### Load model ###
    if LOAD_MODEL:
        actor_model.load_state_dict(torch.load(MODEL_LOAD_PATH_ACTOR))
        critic_model.load_state_dict(torch.load(MODEL_LOAD_PATH_CRITIC))
        print("\n##### Models loaded #####\n")
    
    policy_trainer = PPOTraining(env=env, device=device, actor_model=actor_model, critic_model=critic_model,
                                 actor_optimizer=actor_optimizer, critic_optimizer= critic_optimizer,
                                 ppo_train_epochs=PPO_TRAIN_EPOCHS, ppo_steps=PPO_STEPS, 
                                 ppo_update_epochs=PPO_UPDATE_EPOCHS, gamma=GAMMA, 
                                 gae_lambda=GAE_LAMBDA, epsilon_clip=EPSILON_CLIP, 
                                 entropy_beta=ENTROPY_BETA,
                                 save_interval=SAVE_INTERVAL,
                                 chkpnt_dir=models_dir, fname_batch=fname_batch, 
                                 fname_update=fname_update)
    
    ### Start PPO training ###

    # print("\n\n##### Start Of Training #####\n")
    # policy_trainer.train()
    # print("\n##### End Of Training #####\n\n")

    ### Test trained environment ###
    states, actions, rewards, dist_probs, vol_flows, taus, total_reward, epoch_mean_gamma, sensor_gamma = policy_trainer.test_env()

    # print(f'states: {states}\nactions: {actions}\nrewards: {rewards}\ndist_probs: {dist_probs}\nvol_flows: {vol_flows}\ntaus: {taus}\ntotal_reward: {total_reward}\n')

    # write test data
    fname_test = f'{log_dir}/{TEST_FILE}'
    create_file(fname_test, ["action"] + state_names + ["reward", "dist_prob", "vol_flow"] + tau_state_names + ["epoch_mean_gamma"] + gamma_state_names)

    for step in range(PPO_STEPS):
        policy_test_data = [actions[step]] + states[step].tolist()[0] + rewards[step].tolist()[0] + [dist_probs[step]] + [vol_flows[step]] + taus[step] + [epoch_mean_gamma] + sensor_gamma
        write_file(fname_test, policy_test_data)

    ### Stop environment after training ###
    env.end_measurement()

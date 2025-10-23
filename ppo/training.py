"""
This file contains PPO training algorithm
"""

import time
import torch
import numpy as np

from utils import write_file


class PPOTraining:
    def __init__(self, env, device, actor_model, critic_model, actor_optimizer, 
                 critic_optimizer, ppo_train_epochs, ppo_steps, 
                 ppo_update_epochs, gamma, gae_lambda, epsilon_clip, entropy_beta, 
                 save_interval, chkpnt_dir, fname_batch, fname_update):
        """
        Initializes PPOTraining object.
        """
        self.env = env
        self.device = device

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        
        self.ppo_train_epochs = ppo_train_epochs
        self.ppo_steps = ppo_steps
        self.ppo_update_epochs = ppo_update_epochs
        
        self.mask = torch.cat((torch.ones((self.ppo_steps - 1, 1), 
                                          dtype=torch.float32, 
                                          device=self.device), 
                               torch.zeros((1, 1), dtype=torch.float32, 
                                           device=self.device)))
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.epsilon_clip = epsilon_clip
        self.entropy_beta = entropy_beta
        
        self.chkpnt_dir = chkpnt_dir
        self.save_interval = save_interval
        
        self.fname_batch = fname_batch
        self.fname_update = fname_update
        
    @staticmethod
    def normalize(x):
        """
        Used to normalize the advantage to 0.
        (Theoretically not necessary but decreases variance of advantages 
        and makes convergence more stable and faster in practice)
        """
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def compute_gae_returns(self, next_value, rewards, values):
        """
        Calculates GAE returns (return = GAE advantage + old estimated state value)
        
        GAE Algorithm
        
        1)  mask = 0 if: state = terminal (episode over)
            mask = 1 else
        2)  initialize gae = 0 
            and loop backward from last step in data
        3)  set delta
            delta = reward + gamma * next_state_value * mask - state_value
        4)  update gae
            gae = delta + gammma * lambda * mask * gae
        5)  return for state and action
            return(s,a) = gae + state_value
        6)  reverse the list to returns back to the correct order
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * self.mask[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * self.mask[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def compute_mc_returns(self, rewards):
        """
        Calculates the return of each state using Monte Carlo 
        (easier to understand but not as good as GAE)
        -> just used for debugging and written statistics
        """
        discounted_reward = 0
        returns = []
        for step in reversed(range(len(rewards))):
            discounted_reward = rewards[step] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    def compute_own_returns(self, rewards, next_rewards):
        """
        Own return calculation. Takes for each step the reward plus the next three rewards as return value.
        """
        returns = []
        next_rewards_tensor_list = [torch.tensor(next_reward).unsqueeze(0) for next_reward in next_rewards]
        rewards = rewards + next_rewards_tensor_list

        for step in reversed(range(self.ppo_steps)):
            return_ = torch.stack(rewards[step+1:step+5]).sum(dim=0)
            returns.insert(0, return_)
        return returns

    def ppo_update(self, frame_idx, train_epoch, states, actions, act_log_probs, returns, advantages):
        """
        Function to update the policy and value network.
            1. pass state into networks, obtain predicted actions, values, entropy and new_log_probs
            2. calculate surrogate policy loss and mean squared error value loss
            3. backpropagate the losses through networks using Stochastic Gradient Descent (SGD)
        """
        
        epoch_numbers = []
        epoch_actor_losses = []
        epoch_critic_losses = []
        entropy_losses = []
        epoch_entropies = []

        for update_epoch in range(1, self.ppo_update_epochs + 1):
            
            # print(f'\n##### Update Epoch: {update_epoch} #####\n')
            
            # model prediction of dist and value for all states in batch
            dists = self.actor_model(states)
            values = self.critic_model(states)
            
            entropy = dists.entropy().mean()
            new_log_probs = dists.log_prob(actions)
            
            ### Calculate surrogate losses ###
            ratio = (new_log_probs - act_log_probs).exp()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages

            ### Calculate actor and critic losses ###
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()
            entropy_loss = - self.entropy_beta * entropy
            
            ### Backpropagation ### (old time for one NN ~0.004)
            self.actor_optimizer.zero_grad()
            (actor_loss + entropy_loss).backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ### Store Data ###
            epoch_numbers.append(update_epoch)
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            epoch_entropies.append(entropy.item())

        return epoch_numbers, epoch_actor_losses, epoch_critic_losses, entropy_losses, epoch_entropies

    def collect_batch(self):
        """
        Collects a batch of PPO_STEPS by acting in the environment with current policy
        """
        ### Reset the environment ###
        # state = self.env.reset()
        state, vol_flow, tau = self.env.reset()
        
        ### Batch data ###
        act_log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        dist_probs = []
        vol_flows = []
        taus = []

        for step in range(self.ppo_steps):

            step_start_time = time.perf_counter()  # for fixed time for step

            ### Get action and value prediction ### ~ ??? s (virtual env.)
            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor_model(state)
            value = self.critic_model(state)
            action = dist.sample()
    
            ### Pass action to environment and obtain reward and next state ### ~ 0.0002 s (virtual env.)
            next_state, reward, _, next_vol_flow, next_tau = self.env.step(bool(action.item()))

            ### Get action-log-distribution ### ~ 0.0004 s (virtual env.)
            act_log_prob = dist.log_prob(action)
            
            ### Store data ### ~ 0.0001 s (virtual env.)
            act_log_probs.append(act_log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            states.append(state)
            actions.append(action)
            dist_probs.append(dist.probs)  # (for stats)
            vol_flows.append(vol_flow)
            taus.append(tau)

            vol_flow = next_vol_flow
            tau = next_tau

            state = next_state

            ### Ensure step takes at least 0.005 s ###
            while(time.perf_counter() - step_start_time) < 0.005:
                pass

        ### get three additional rewards for own returns
        next_rewards = []

        for _ in range(5):
            step_start_time = time.perf_counter()  # for fixed time for step

            ### Get action and value prediction ### ~ ??? s (virtual env.)
            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor_model(state)
            action = dist.sample()

            ### Pass action to environment and obtain reward and next state ### ~ 0.0002 s (virtual env.)
            next_state, reward, _, _, _ = self.env.step(bool(action.item()))

            next_rewards.append(reward)
            state = next_state

            ### Ensure step takes at least 0.005 s ###
            while (time.perf_counter() - step_start_time) < 0.005:
                pass

        # open valves after batch collection (for training with constant volume flow)
        self.env.sensors.open_valve()
        # self.env.sensors.close_valve()

        return act_log_probs, values, states, actions, rewards, dist_probs, next_state, next_rewards, vol_flows, taus

    def calc_batch_gamma(self, taus):
        """
        Calculates the forward flow fraction of a batch for each sensor
        -> used for written statistics
        """
        num_sensors = len(taus[0])
        gamma = []  # list with gamma of each sensor
        for idx in range(num_sensors):
            sensor_values = [sublist[idx] for sublist in taus]
            pos_count = sum(1 for value in sensor_values if value > 0)
            sensor_gamma = pos_count/self.ppo_steps
            gamma.append(sensor_gamma)
            
        return gamma

    def test_env(self):
        # not tested jet
        """
        Test the total reward of a trained model
        for PPO_STEPS steps in the environment
        """
        ### Test data ###
        states    = []
        actions   = []
        rewards   = []
        dist_probs = []
        vol_flows = []
        taus = []

        state, vol_flow, tau = self.env.reset()
        total_reward = 0

        for _ in range(self.ppo_steps):

            step_start_time = time.perf_counter()  # for fixed time for step

            state = torch.FloatTensor(state).to(self.device)
            dist = self.actor_model(state)
            # value = self.critic_model(state)
            # action = dist.sample()
            dist_prob = dist.probs.item()
            action = round(dist_prob)
            next_state, reward, _, next_vol_flow, next_tau = self.env.step(bool(action))
            # reward = reward.item()

            ### Store data ###
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            states.append(state)
            actions.append(action)
            dist_probs.append(dist_prob)
            vol_flows.append(vol_flow)
            taus.append(tau)

            vol_flow = next_vol_flow
            tau = next_tau
            state = next_state

            total_reward += reward

            ### Ensure step takes at least 0.005 s ###
            while(time.perf_counter() - step_start_time) < 0.005:
                pass

        sensor_gamma = self.calc_batch_gamma(taus)
        epoch_mean_gamma = np.mean(sensor_gamma)

        return states, actions, rewards, dist_probs, vol_flows, taus, total_reward, epoch_mean_gamma, sensor_gamma

    def train(self):
        """
        Main PPO training loop 
        with batch colletion and network update for PPO_TRAIN_EPOCHS epochs
        """
        frame_idx = 0

        ### Training ###

        for train_epoch in range(1, self.ppo_train_epochs + 1):

            print(f'\n##### Training Epoch: {train_epoch} #####\n')

            ### Collect new batch ###
            act_log_probs, values, states, actions, rewards, dist_probs, next_state, next_rewards, vol_flows, taus = self.collect_batch()

            # time duration of update and write time
            update_and_write_start_time = time.perf_counter()

            ### Calculate returns for the batch using GAE ###
            next_state = torch.FloatTensor(next_state).to(self.device) # convert state to torch tensor
            next_value = self.critic_model(next_state)  # get model value prediction

            gae_returns = self.compute_gae_returns(next_value, rewards, values)
            mc_returns = self.compute_mc_returns(rewards)  # monte carlo returns (just for stats)
            own_returns = self.compute_own_returns(rewards, next_rewards)

            ### Detach and concatenat tensor lists ###
            rewards = torch.cat(rewards).detach()
            act_log_probs = torch.cat(act_log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            gae_returns = torch.cat(gae_returns).detach()
            mc_returns = torch.cat(mc_returns).detach()
            own_returns = torch.cat(own_returns).detach()
            dist_probs = torch.cat(dist_probs).detach()

            ### Calculate advantage for the batch ###
            gae_advantages = gae_returns - values
            gae_advantages = self.normalize(gae_advantages)
            
            mc_advantages = mc_returns - values  # monte carlo advantage (just for stats)
            mc_advantages = self.normalize(mc_advantages)

            own_advantages = own_returns - values
            own_advantages = self.normalize(own_advantages)

            # print(f'actions: {actions}')
            
            ### Calculate total reward of current batch ### (for stats)
            epoch_total_reward = sum(rewards).item()
            
            ### Calculate gamma ### (for stats)
            sensor_gamma = self.calc_batch_gamma(taus)
            epoch_mean_gamma = np.mean(sensor_gamma)

            ### Calculate number of states where valve was opened ###
            count_valve_open = sum(actions).item()
            
            ### Write PPO Batch data to file ### ~ 0.005 s
            for step in range(self.ppo_steps):

                frame_idx += 1
                state_elements = [s.item() for s in states[step]]
                action_elements = [a.item() for a in actions[step]]

                ppo_batch_data = [frame_idx, train_epoch, step + 1, 
                                  epoch_total_reward,
                                  gae_returns[step].item(),
                                  mc_returns[step].item(),
                                  own_returns[step].item(),
                                  values[step].item(),
                                  rewards[step].item(), 
                                  gae_advantages[step].item(), 
                                  mc_advantages[step].item(),
                                  own_advantages[step].item(),
                                  act_log_probs[step].item(), 
                                  dist_probs[step].item()] + action_elements + [count_valve_open] + state_elements + \
                                 [vol_flows[step]] + taus[step] + [epoch_mean_gamma] + sensor_gamma

                write_file(self.fname_batch, ppo_batch_data)

            ### Optimizing the policy and value network ###
            update_numbers, update_actor_losses, update_critic_losses, update_entropy_losses, update_entropies = self.ppo_update(frame_idx, train_epoch, states,
                                              actions, act_log_probs, 
                                              own_returns, own_advantages)
            
            ### Write PPO Update data to file ###
            for epoch in range(len(update_numbers)):
                ppo_update_data = [frame_idx, train_epoch, update_numbers[epoch], update_actor_losses[epoch], update_critic_losses[epoch], update_entropy_losses[epoch], update_entropies[epoch]]
                write_file(self.fname_update, ppo_update_data)
            
            ### Save models ###
            if train_epoch % self.save_interval == 0:
                torch.save(self.actor_model.state_dict(), f'{self.chkpnt_dir}/epoch_{train_epoch}_torch_actor_model')
                torch.save(self.critic_model.state_dict(), f'{self.chkpnt_dir}/epoch_{train_epoch}_torch_critic_model')
                print("\n##### Models saved #####\n")
            
            print(f'\nTraining Epoch: {train_epoch} \nTotal Reward of current Batch: {sum(rewards).item()} \nBatchsize: {self.ppo_steps}\nEpoch mean actor losses: {np.mean(update_actor_losses)}\nEpoch mean critic losses: {np.mean(update_critic_losses)}\nEpoch mean entropy: {np.mean(update_entropies)}\nMean Forward Flow Fraction: {epoch_mean_gamma}\nCount valve open: {count_valve_open}\n')

            # time duration of update and write time
            # print(f'update_and_write_start_time: {time.perf_counter() - update_and_write_start_time}')

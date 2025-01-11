import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from harmonization_env.harmonizer.enrivonment import HarmonizationEnv
from harmonization_env.harmonizer.batcher import Batcher
from harmonization_env.harmonizer.utils import MelodySampler


class Agent:

  def __init__(self, 
               env : HarmonizationEnv, 
               net : nn.Module, 
               **kwargs):

    self.env = env
    self.device = self.env.device

    self.net = net
    self.net = net.to(self.device)
  
    self.batch_size = kwargs.get('batch_size', 2048)
  
    self.gamma = kwargs.get('gamma', 0.99)
    self.lambda_ = kwargs.get('lambda_', 0.95)
    self.lr = kwargs.get('lr', 1e-5)
    self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.net.parameters(), lr = self.lr))

    self.epsilon = kwargs.get('epsilon', 0.2)

    self.actor_weight = kwargs.get('actor_weight', 1.0)
    self.critic_weight = kwargs.get('critic_weight', 0.5)
    self.entropy_weight = kwargs.get('entropy_weight', 0.15)
    self.batcher = kwargs.get('batcher', None)

    self.load_ = kwargs.get('load', False)
    self.path = self.net.path # self.path = kwargs.get('path', 'net_mlp.pth')

    self.sampler = MelodySampler(device = self.device)

    if self.batcher is not None: self.batcher = self.batcher 

    else: self.batcher = Batcher(device = self.device, 
                                env = self.env, 
                                net = self.net, 
                                gamma = self.gamma, 
                                lambda_ = self.lambda_, 
                                batch_size = self.batch_size)

    if self.load_: self.load()


  def train(self, 
            num_iterations : int = 1000, 
            iter_per_batch : int = 5, 
            check_step : int = 10, 
            fine_tune : bool = False,
            save : bool = True,
            print_out : bool = False,
            path : str = None):

    if path == None: path = self.path
    if num_iterations < check_step: check_step = 1

    iterator = range(num_iterations) if print_out else tqdm(range(num_iterations))

    for num_iter in iterator:

      # self.env.reset()
      if fine_tune: self.env.reset()
      else: self.env.reset(melody = self.sampler.get(batch_size = self.batcher.batch_size))

      timesteps = self.env.melody.shape[-1]

      data = self.batcher.get()
      for key, value in data.items(): setattr(self, key, value)

      avg_loss = 0

      for _ in range(iter_per_batch):

        new_probs = torch.zeros((self.batch_size, timesteps), dtype = torch.float32, device = self.device)
        critic_values = torch.zeros((self.batch_size, timesteps), dtype = torch.float32, device = self.device)
        avg_entropies = torch.zeros(timesteps - 1, dtype = torch.float32, device = self.device)

        self.optimizer.zero_grad()

        for k in range(1, timesteps):

            new_dist, new_values = self.net(self.states[:, k - 1, :])
            critic_values[:, k] = new_values.squeeze()

            new_probs[:, k - 1] = new_dist[torch.arange(self.batch_size), self.actions[:, k - 1]]

            entropy = -torch.sum(new_dist * torch.log(new_dist + 1e-10), dim = 1)
            avg_entropies[k - 1] = entropy.mean()

        ratio = new_probs / (self.actions_prob + 1e-5)
        ratio[:, -1] = 1

        weighted_probs = self.advantages * ratio
        weighted_clipped_probs = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantages

        actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).sum(dim = 1).mean()
        critic_loss = F.mse_loss(critic_values, self.returns, reduction = 'none').sum(dim = 1).mean()
        entropy_loss = - avg_entropies.sum()

        total_loss = self.actor_weight * actor_loss + self.critic_weight * critic_loss + self.entropy_weight * entropy_loss

        avg_loss += total_loss
        total_loss.backward(retain_graph = True)

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm = 0.5)

        self.optimizer.step()

      if num_iter % check_step == 0: 
        if print_out: print(f'n_iter: {num_iter + 1}/{num_iterations}. avg loss: {avg_loss:.3f}')
        if save: 
          print(f'saved at {self.path}')
          self.save()
    
  def get(self, 
          melody : torch.Tensor,
          fine_tune : bool = True,
          num_iterations : int = 100,
          num_samples : int = 10000):

    self.env.reset(melody = melody)
    if fine_tune: self.train(num_iterations = num_iterations, fine_tune = True, save = False)

    batch_size = self.batcher.batch_size
    self.batcher.batch_size = num_samples
    
    data = self.batcher.get()
    for key, value in data.items(): setattr(self, key, value)

    self.batcher.batch_size = batch_size

    total_rewards = torch.sum(self.rewards, dim = 1)
    index = torch.argmax(total_rewards).item()

    return self.states[index, 1 :, 0].to(dtype = torch.int32), self.rewards[index, 1 :], 


  def save(self, path : str = None):
    
    if path == None: path = self.path 
    torch.save(self.net.state_dict(), path)


  def load(self):

      net_dict = torch.load(self.path, weights_only = False, map_location = torch.device('cpu'))
      self.net.load_state_dict(net_dict)

'''
      if not fine_tune: 
        
        if (num_iter - 1) % n == 0 and num_iter > 0:
          avg_score = 0

          for _ in range(50):
            self.env.reset(melody = melody_sampler())
            _, _ ,_ , _ , rewards = self.generate_batch(batch_size = 32)
            avg_score += rewards.sum() / 32
          
          avg_score /= 50

          if not fine_tune: print(f"avg_score {round(avg_score.item(), 3)}.")

          if running_best < avg_score:
            running_best = avg_score
            self.save_model()

          else: self.load_model()
      
      if fine_tune: self.save_model()

  def save_model(self): torch.save(self.net.state_dict(), self.path)

  def load_model(self):

      net_dict = torch.load(self.path, weights_only = False, map_location = torch.device('cpu'))
      self.net.load_state_dict(net_dict)

  def __call__(self, melody, num_samples = 1, test_samples = 1000, print_ = True, save = True, entropy_weight = 0.3, filename = 'data'):

    self.env.reset(melody = melody)

    old_entropy = self.entropy_weight
    self.entropy_weight = entropy_weight

    path_og = self.path
    self.path = f"{self.path}_tmp"

    shutil.copy(path_og, self.path)
    self.train(num_iterations = 100, batch_size = 64, iter_per_batch = 2, check_step = 1, fine_tune = True)

    if num_samples > test_samples: test_samples = num_samples

    states, _ ,_ , _ , rewards = self.generate_batch(batch_size = test_samples)
    scores = rewards.sum(dim = 1)
    avg_score = (rewards.sum() / test_samples).item()
    
    _, indices = torch.topk(scores, num_samples)

    print(f"avg_score over {test_samples} test samples: {round(avg_score, 3)}. Top {num_samples} samples:")

    if print_: print(f'Melody: {self.env.melody}')
    if print_: print('--------------------------------')

    get_chord = self.env.H.chord_tags
    best_chords = []

    for k in range(num_samples):

        chords = [get_chord[int(states[indices[k], i, 0])] for i in range(states.shape[1])]

        best_chords.append(chords)
        if print_: print(chords)

        if print_: print(f'{rewards[indices[k], :]}. Total reward: {torch.sum(rewards[indices[k],:])}')
        if print_: print('--------------------------------')

    if save: self.save_data(best_chords, filename)

    os.remove(self.path)
    self.path = path_og

    self.entropy_weight = old_entropy

  def save_data(self, chords, filename = 'data'):

    filename = filename + '.json'

    melody_list = self.env.melody.int().tolist()
    
    data = {
        "melody": melody_list,  # Melody as a list of integers
        "chords": chords        # The list of list of strings
    }
        
    with open(filename, "w") as f: json.dump(data, f, indent = 4)
    print(f"{filename} saved successfully.")


if __name__ == '__main__': pass'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from harmonization_env.resources import get_resource_file



def pick_type(preset : int):

    if preset == 'mlp_test':

        shared_hidden_dims = [16, 16, 16, 32, 32, 64, 128, 256]
        actor_hidden_dims = [128, 64, 32, 32, 32]
        critic_hidden_dims = [128, 64, 32, 16, 8, 8, 4, 4]
    
    return shared_hidden_dims, actor_hidden_dims, critic_hidden_dims


class Net(nn.Module):

    def __init__(self, 
                 input_dim : int = 3, 
                 preset : str = 'mlp_test', 
                 shared_hidden_dims : list = [16, 32, 64, 32], 
                 actor_hidden_dims : list = [64, 32],
                 critic_hidden_dims : list = [64, 32],
                 actor_output_dim : list = 10, 
                 critic_output_dim : list = 1, 
                 device : str = 'cpu',
                 path = None):
        
        if path is None: shared_hidden_dims, actor_hidden_dims, critic_hidden_dims = pick_type(preset)
        else: preset = False

        super(Net, self).__init__()

        self.shared_mlp = self._create_mlp(input_dim, shared_hidden_dims)

        self.actor_mlp = self._create_mlp(shared_hidden_dims[-1], actor_hidden_dims)
        self.actor_head = nn.Linear(actor_hidden_dims[-1], actor_output_dim)

        self.critic_mlp = self._create_mlp(shared_hidden_dims[-1], critic_hidden_dims)
        self.critic_head = nn.Linear(critic_hidden_dims[-1], critic_output_dim)

        if path is not None: self.path = path
        else: self.path = None

        if preset is not None and preset is not False:
            if preset == 'mlp_test': 
                self.path = get_resource_file(f"{preset}.pth")
                state_dict = torch.load(self.path, map_location = torch.device(device), weights_only = False)

            self.load_state_dict(state_dict)


    def _create_mlp(self, input_dim, hidden_dims):

        layers = []
        in_features = input_dim

        for hidden_dim in hidden_dims:

            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            ])

            in_features = hidden_dim

        return nn.Sequential(*layers)
    

    def forward(self, x):
        
        shared_features = self.shared_mlp(x)
        
        actor_features = self.actor_mlp(shared_features)
        actor_output = self.actor_head(actor_features)
        actor_output = F.softmax(actor_output, dim = -1)
        
        critic_features = self.critic_mlp(shared_features)
        critic_output = self.critic_head(critic_features)
        
        return actor_output, critic_output
    

    '''def save(self, path = None): 

        if path == None: path = self.path
        torch.save(self.state_dict(), path)


    def load(self, path):
        
        if path == None: path = self.path
        net_dict = torch.load(path, weights_only = False, map_location = torch.device('cpu'))
        self.net.load_state_dict(net_dict)'''
# Music Harmonization Package

A Reinforcement Learning package to generate musical harmonies, starting from a general sequence of notes in the scale of C major. Outputs musical harmonies, MIDI files, and mp4 videos.

## Prerequisites

- **ffmpeg** must be installed on your system.
  - [Download ffmpeg](https://ffmpeg.org/download.html)

## Basic Usage

Here's a complete example showing the main features:

```python
import torch
from harmonization_env import HarmonizationEnv, Net, Agent, Voicer, MIDIGenerator, VideoGenerator

# Set device for computations
device = 'cpu'

# Create a melody sequence
# Notes are represented as MIDI numbers (60 = middle C)
melody = torch.tensor([72, 74, 76, 77, 79, 81, 83, 84], dtype=torch.int32)  # C major scale

# Initialize the harmonization environment
env = HarmonizationEnv(melody, device)

# Load neural network model
net = Net(preset='mlp_test')
# It is also possible to create your own network (MLP for now) by specifying parameters:
# custom_net = Net(shared_hidden_dims=[...], 
#                  actor_hidden_dims=[...],
#                  critic_hidden_dims=[...],
#                  path='path_to_custom_net.pth')

# Create and train the harmonization agent
agent = Agent(env, net, batch_size=2048, lr=1e-5)
agent.train(num_iterations=1000, check_step=20, save=True, print_out=True)

# Get chords (and relative reward). fine_tune=True allows the net to train directly on the specific melody input.
chords, reward = agent.get(melody=melody, 
                          fine_tune=True, 
                          num_iterations=100)

# Create voicing
v = Voicer(melody, chords)
voices = v.get()

# Generate MIDI file
player = MIDIGenerator()
player.generate(voices, filenames='test0.mid')
player.play(filename='test0.mid')

# Create visualization video
# Download soundfont from https://member.keymusician.com/Member/FluidR3_GM/index.html
vg = VideoGenerator('test0.mid', soundfont_path='FluidR3_GM.sf2')
vg.get_video(output_filename='output.mp4')
```
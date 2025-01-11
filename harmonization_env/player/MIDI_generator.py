import torch
import pygame
from midiutil import MIDIFile

class MIDIGenerator:

    def __init__(self, 
                 tempo: int = 60, 
                 volume: int | list[int] = 100, 
                 channel: int = 0):
        
        self.tempo = tempo
        self.volume = volume if isinstance(volume, list) else [volume]
        self.channel = channel
        
        pygame.mixer.init()

    def generate(self, 
                voices: torch.Tensor, 
                filenames: str | list[str] = "output.mid", 
                durations: torch.Tensor = None) -> str:
        
        if voices.ndim == 2: voices = voices.unsqueeze(0)
        if isinstance(filenames, str): filenames = [filenames]
        
        if durations is None:

            self.durations = torch.ones_like(voices, dtype = torch.float)
            self.durations[:, :, -1] = 4.0

        else: self.durations = durations

        for i in range(voices.shape[0]): 

            midi = self.get(voices[i, :, :], self.durations[i, :, :])
            with open(filenames[i], "wb") as outf: midi.writeFile(outf)
    
    def get(self, 
            voices: torch.Tensor,
            durations: torch.Tensor,
            enhance_melody: bool = True) -> MIDIFile:
        
        num_tracks = voices.shape[0]
        midi = MIDIFile(num_tracks, adjust_origin=True)
    
        for track in range(num_tracks):

            midi.addTempo(track, 0, self.tempo)

            if enhance_melody and track == 0: volume = min(127, self.volume[0] * 2)
            else: volume = self.volume[0] if len(self.volume) == 1 else self.volume[track]

            time = 0.0

            for i in range(voices.shape[1]):
                pitch = int(voices[track, i].item())
                duration = float(durations[track, i].item())

                # Add note if it's not a rest (-1)
                if pitch != -1:
                    midi.addNote(
                        track = 0, # track,
                        channel = self.channel,
                        pitch = pitch,
                        time = time,
                        duration = duration,
                        volume = volume
                    )

                time += duration

        return midi
    
    def play(self, filename: str):

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy(): pygame.time.wait(100)

        except Exception as e: print(f"Error playing MIDI file: {e}")

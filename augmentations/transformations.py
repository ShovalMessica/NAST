import random
from augmentations.audio_transformations import InjectNoise, TimeWarp, PitchShift, AddReverb


class AudioAugmentations:
    def __init__(self, config, phase):
        self.phase = phase
        self.config = config

        self.trans_noise_5_30 = InjectNoise(
            paths=self.config['augmentations']['noise_paths'],
            sr=self.config['audio']['sample_rate'],
            noise_levels=(5, 30),
            probability=1
        )

        self.trans_time_stretch = TimeWarp(
            rates=(self.config['augmentations']['time_warp']['min_rate'],
                   self.config['augmentations']['time_warp']['max_rate']),
            probability=1.0
        )

        self.trans_pitch_shift = PitchShift(
            sr=self.config['audio']['sample_rate'],
            n_steps=self.config['augmentations']['pitch_shift']['n_steps'],
            probability=1.0
        )

        self.trans_reverberation = AddReverb(
            probas=1,
            strong=self.config['augmentations']['reverb']['strong']
        )

        self.augmentations = [
            self.trans_noise_5_30,
            self.trans_time_stretch,
            self.trans_pitch_shift,
            self.trans_reverberation
        ]

    def augment(self, x):
        if self.phase == 'phase1':
            if random.random() >= 0.5:
                return x
        elif self.phase == 'phase2':
            if random.random() >= 1.0:
                return x

        return random.choice(self.augmentations)(x)

"""Sub-package containing useful transforms for audio data."""
from __future__ import division

import json
import types
import logging
import random
import librosa
import numpy as np
import torch
import os
from augmentations.rev_room import RevRoom
from augmentations.reverb_sampler import draw_params

logger = logging.getLogger(__name__)


###############################
# General/Abstract Transforms #
###############################
class Transform(object):
    """Abstract base class for a transform.
    All other transforms should subclass it. All subclassses should
    override __call__ and __call__ should expect one argument, the
    data to transform, and return the transformed data.
    Any state variables should be stored internally in the __dict__
    class attribute and accessed through self, e.g., self.some_attr.
    """

    def __call__(self, data):
        raise NotImplementedError


class Compose(Transform):
    """Defines a composition of transforms.
    Data will be processed sequentially through each transform.
    Attributes:
        transforms: A list of Transforms to apply sequentially.
    """

    def __init__(self, transforms):
        """Inits Compose with the specified transforms.
        Args:
            transforms: Any iterable of transforms.
        """
        self._transforms = list(transforms)

    def __len__(self):
        return len(self._transforms)

    def __getitem__(self, index):
        return self._transforms[index]

    def __call__(self, data):
        for transform in self._transforms:
            data = transform(data)
        return data


class ToTensor(Transform):
    """Converts a numpy.ndarray to a torch.*Tensor."""

    def __call__(self, nparray):
        return torch.from_numpy(nparray)


class ToArray(Transform):
    """Converts a torch.*Tensor to a numpy.ndarray."""

    def __call__(self, tensor):
        return tensor.numpy()


class Lambda(Transform):
    """Applies a lambda as a transform.
    Attributes:
        func: A lambda function to be applied to data.
    """

    def __init__(self, func):
        """Inits Lambda with func."""
        assert isinstance(func, types.LambdaType)
        self.func = func

    def __call__(self, data):
        return self.func(data)


#############################
# Audio-oriented transforms #
#############################
class TimeWarp(Transform):
    """Time-stretch an audio series by a fixed rate with some probability.
    The rate is selected randomly from between two values.
    It's just a jump to the left...
    """

    def __init__(self, rates=(0.7, 1.0), probability=0.0):
        self.rates = rates
        self.probability = probability

    def __call__(self, y):
        success = np.random.binomial(1, self.probability)
        if success:
            rate = np.random.uniform(*self.rates)
            return librosa.effects.time_stretch(y=y, rate=rate)
        return y


class PitchShift(Transform):
    """Pitch-shift the waveform by n_steps half-steps."""

    def __init__(self, sr=8000, n_steps=1, probability=0.0):
        self.sr = sr
        self.n_steps = n_steps
        self.probability = probability

    def __call__(self, y):
        success = np.random.binomial(1, self.probability)
        if success:
            y_hat = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=self.n_steps)
            return y_hat / np.abs(y_hat).max()
        return y


class InjectNoise(Transform):
    """Adds noise to an input signal with some probability and some SNR.
    Only adds a single noise file from the list right now.
    """

    def __init__(self,
                 path=None,
                 paths=None,
                 sr=8000,
                 noise_levels=(-5, 30),
                 probability=0.4):
        if path:
            if not os.path.exists(path):
                logger.error("Directory doesn't exist: {}".format(path))
                raise IOError
            # self.paths = librosa.util.find_files(path)
            with open(path, 'r') as f:
                self.paths = json.load(f)
                self.paths = [item[0] for item in self.paths]
        else:
            self.paths = paths
        self.sr = sr
        self.noise_levels = noise_levels
        self.probability = probability
        self.EPS = 1e-8

    # Function to mix clean speech and noise at various SNR levels
    def _snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean ** 2).mean() ** 0.5
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean
        rmsclean = (clean ** 2).mean() ** 0.5

        rmsnoise = (noise ** 2).mean() ** 0.5

        # The file is too silent to be added as noise. Returning the input unchanged.
        if rmsnoise < self.EPS:
            return clean, clean, clean

        scalarnoise = 10 ** (-25 / 20) / rmsnoise
        noise = noise * scalarnoise
        rmsnoise = (noise ** 2).mean() ** 0.5

        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10 ** (snr / 20)) / (rmsnoise + self.EPS))
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        return clean, noisenewlevel, noisyspeech

    def __call__(self, data):
        success = np.random.binomial(1, self.probability)
        if success:
            random_file = random.choice(os.listdir(self.paths))
            random_file = os.path.join(self.paths, random_file)
            # random.choice(os.listdir(self.paths))
            noise_src, _ = librosa.load(
                random_file, sr=self.sr)
            noise_offset_fraction = np.random.rand()
            noise_level = np.random.uniform(*self.noise_levels)

            noise_dst = np.zeros_like(data)

            src_offset = int(len(noise_src) * noise_offset_fraction)
            src_left = len(noise_src) - src_offset

            dst_offset = 0
            dst_left = len(data)

            while dst_left > 0:
                copy_size = min(dst_left, src_left)
                np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                          noise_src[src_offset:src_offset + copy_size])
                if src_left > dst_left:
                    dst_left = 0
                else:
                    dst_left -= copy_size
                    dst_offset += copy_size
                    src_left = len(noise_src)
                    src_offset = 0
            _, _, noisy_rec = self._snr_mixer(data, noise_dst, noise_level)
            return noisy_rec
        return data

    def __getitem__(self, index):
        return self.paths[index]


class SwapSamples(Transform):
    """Intentionally corrupts mono audio by swapping adjacent samples.
    Iterates over each sample (excluding the boundary) of the input
    signal and will stochastically swap the sample with its neighbor.
    Whether the neighbor is the left or right sample is chosen randomly
    with equal probability.
    """

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, y):
        # Vectorize this?
        for i in range(1, len(y) - 1):
            success = np.random.binomial(1, self.probability)
            if success:
                swap_right = np.random.binomial(1, 0.5)
                if swap_right:
                    y[i], y[i + 1] = y[i + 1], y[i]
                else:
                    y[i], y[i - 1] = y[i - 1], y[i]
        return y


class TimeShift(Transform):
    """Stochastically shifts audio samples."""

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, y):
        success = np.random.binomial(1, self.probability)
        if success:
            min_max_displacement = [0.1 * y.shape[0], 0.15 * y.shape[0]]
            shift = np.random.randint(*min_max_displacement)
            if np.random.binomial(1, 0.5):  # shift to the right
                y[shift:] = y[:-shift]
                y[:shift] = 0
            else:
                y[:-shift] = y[shift:]
                y[-shift:] = 0
        return y / np.abs(y).max()


class AddReverb(Transform):
    """Adds reverb to an existing recording by simulating different room types.
    """

    def __init__(self, probas=1, mic_distance=4, strong=False):
        self.probas = probas
        self.mic_distance = mic_distance
        self.strong = strong
        self.reverb_levels = {0: "low", 1: "medium", 2: "high"}

    def __call__(self, mix):
        if random.random() > self.probas:
            return mix
        # randomly select the reverb level and the mic to use
        if self.strong:
            rev_level_i = np.random.choice(3, p=[.5, .3, .2])
        else:
            rev_level_i = np.random.choice(3, p=[.85, .1, .05])
        mic_idx = np.random.choice(2)
        rev_level = self.reverb_levels[rev_level_i]

        # buid the room
        room_params = draw_params(self.mic_distance, rev_level)
        room = RevRoom([room_params[0][0], room_params[0][1], room_params[0][2]],
                       [[room_params[1][0][0], room_params[1][0][1], room_params[1][0][2]],
                        [room_params[1][1][0], room_params[1][1][1], room_params[1][1][2]]],
                       [room_params[2][0], room_params[2][1], room_params[2][2]],
                       room_params[3])
        room.generate_rirs()
        room.add_audio(mix)
        reverberant = room.generate_audio()
        if np.abs(reverberant).max() < 1e-8:
            return (reverberant[0, mic_idx] / (1e-8)).astype(np.float32)
        return (reverberant[0, mic_idx] / np.abs(reverberant).max()).astype(np.float32)


# test
if __name__ == "__main__":
    rev = AddReverb(rev_clean=1)
    y, sr = librosa.load('clean.wav', sr=None)
    y_hat = rev(y)
    print(y.std(), y_hat.std())
    le = y.shape[-1]
    m = 0
    md = 0
    for d in range(1, 4096):
        dot = (y[d:] * y_hat[:le - d]).mean()
        if dot > m:
            m = dot
            md = d
    print(m, md)

    librosa.output.write_wav('rev.wav', y_hat, sr)
    import sys

    sys.exit(0)

    # Test TimeShift augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([TimeShift(probability=1.0)])
    y_hat = trans(y)
    librosa.output.write_wav('time_shift.wav', y_hat, sr)

    # Test PitchShift augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([PitchShift(probability=1.0)])
    y_hat = trans(y)
    librosa.output.write_wav('pitch_shift.wav', y_hat, sr)

    # Test TimeWarp augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([TimeWarp(probability=1.0)])
    y_hat = trans(y)
    librosa.output.write_wav('time_warp.wav', y_hat, sr)

    # Test SwapSamples augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([SwapSamples(probability=1.0)])
    y_hat = trans(y)
    librosa.output.write_wav('swap_samples.wav', y_hat, sr)

    # Test NoiseInject augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([InjectNoise(path='noises', probability=1.0)])
    y_hat = trans(y)
    librosa.output.write_wav('inject.wav', y_hat, sr)

    # Test NoiseInject augmentation
    y, sr = librosa.load('1.wav', sr=None)
    trans = Compose([AddReverb()])
    y_hat = trans(y)
    librosa.output.write_wav('rev.wav', y_hat, sr)

import numpy as np
import math
import torch.nn as nn
from pyroomacoustics.room import SoundSource
from pyroomacoustics.parameters import constants
from pyroomacoustics.directivities import Directivity
from fairseq.models.wav2vec.wav2vec2 import SamePad
import fairseq.models.wav2vec.wav2vec2 as wav2vec2


def modified_make_conv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    if not is_batch_norm:
        pos_conv = nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(k), nn.GELU())

    return pos_conv


wav2vec2.make_conv_pos = modified_make_conv_pos


def modified_get_rir(self, mic, visibility, Fs, t0=0.0, t_max=None):
    """
    Compute the room impulse response between the source
    and the microphone whose position is given as an
    argument.

    Parameters
    ----------
    mic: ndarray
        microphone position
    visibility: int32
        1 if mic visibile from source, 0 else. Exact type is important for C extension
    Fs: int
        sampling frequency
    t0: float
        time offset, defaults to 0
    t_max: None
        max time, defaults to 1.05 times the propagation time from mic to source
    """

    # fractional delay length
    fdl = constants.get("frac_delay_length")
    fdl2 = (fdl - 1) // 2

    # compute the distance
    dist = self.distance(mic)
    time = dist / constants.get("c") + t0
    if self.damping.shape[0] == 1:
        alpha = self.damping[0, :] / (4.0 * np.pi * dist)
    else:
        raise NotImplementedError("Not implemented for multiple frequency bands")

    # the number of samples needed
    if t_max is None:
        # we give a little bit of time to the sinc to decay anyway
        N = np.ceil((1.05 * time.max() - t0) * Fs)
    else:
        N = np.ceil((t_max - t0) * Fs)

    N += fdl

    t = np.arange(N) / float(Fs)
    ir = np.zeros(t.shape)

    try:
        # Try to use the Cython extension
        from pyroomacoustics.build_rir import fast_rir_builder

        time_adjust = time + fdl2 / Fs
        fast_rir_builder(ir, time_adjust, alpha, visibility.astype(np.int32), Fs, fdl)

    except ImportError:
        print("Cython-extension build_rir unavailable. Falling back to pure python")
        # fallback to pure Python implemenation
        from pyroomacoustics.utilities import fractional_delay

        for i in range(time.shape[0]):
            if visibility[i] == 1:
                time_ip = int(np.round(Fs * time[i]))
                time_fp = (Fs * time[i]) - time_ip
                ir[time_ip - fdl2: time_ip + fdl2 + 1] += alpha[
                                                              i
                                                          ] * fractional_delay(time_fp)

    return ir


SoundSource.get_rir = modified_get_rir

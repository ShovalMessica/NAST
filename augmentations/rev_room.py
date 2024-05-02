import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.parameters import constants


class RevRoom(pra.room.ShoeBox):

    def __init__(self, p, mics, s1, T60, fs=16000,
                 t0=0., sigma2_awgn=None, s2=None):

        self.T60 = T60
        self.max_rir_len = np.ceil(T60*fs).astype(int)

        volume = p[0]*p[1]*p[2]
        surface_area = 2*(p[0]*p[1] + p[0]*p[2] + p[1]*p[2])
        absorption = 24 * volume * np.log(10.0) / (constants.get('c') * surface_area * T60)

        # minimum max order to guarantee complete filter of length T60
        max_order = np.ceil(T60 * constants.get('c') / min(p)).astype(int)

        super().__init__(p, fs=fs, t0=t0, absorption=absorption,
                         max_order=max_order, sigma2_awgn=sigma2_awgn,
                         sources=None, mics=None)

        self.add_source(s1)
        if s2 is not None:
            self.add_source(s2)

        self.add_microphone_array(pra.MicrophoneArray(np.array(mics).T, fs))

    def add_audio(self, s1, s2=None):
        self.sources[0].add_signal(s1)
        if s2 is not None:
            self.sources[1].add_signal(s2)

    def compute_rir(self):

        self.rir = []
        self.visibility = None

        self.image_source_model()

        for m, mic in enumerate(self.mic_array.R.T):
            h = []
            for s, source in enumerate(self.sources):
                h.append(source.get_rir(
                    mic, self.visibility[s][m], self.fs, self.t0)[:self.max_rir_len])
            self.rir.append(h)

    def generate_rirs(self):

        original_max_order = self.max_order
        self.max_order = 0

        self.compute_rir()

        self.rir_anechoic = self.rir

        self.max_order = original_max_order

        self.compute_rir()

        self.rir_reverberant = self.rir

    def generate_audio(self, anechoic=False):

        if not self.rir:
            self.generate_rirs()
        if anechoic:
            self.rir = self.rir_anechoic
        else:
            self.rir = self.rir_reverberant
        audio_array = self.simulate(return_premix=True, recompute_rir=False)
        return audio_array

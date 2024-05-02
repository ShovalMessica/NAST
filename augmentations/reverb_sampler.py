import numpy as np
from numpy.random import uniform

def draw_params(mic_width, reverb_level):

    room_dim = np.array([uniform(5, 10),
                         uniform(5, 10),
                         uniform(3, 4)])

    center = np.array([room_dim[0]/2 + uniform(-0.2, 0.2),
                       room_dim[1]/2 + uniform(-0.2, 0.2),
                       uniform(0.9, 1.8)])

    mic_theta = uniform(0, 2*np.pi)
    mic_offset = np.array([np.cos(mic_theta) * mic_width/2,
                           np.sin(mic_theta) * mic_width/2,
                           0])
    mics = np.array([center + mic_offset,
                     center - mic_offset])

    s_dist = uniform(0.66, 2)
    s_theta = uniform(0, 2*np.pi)
    s_height = uniform(0.9, 1.8)
    s_offset = np.array([np.cos(s_theta) * s_dist,
                          np.sin(s_theta) * s_dist,
                          s_height - center[2]])
    s = center + s_offset

    if reverb_level == "high":
        T60 = uniform(0.4, 1.0)
    elif reverb_level == "medium":
        T60 = uniform(0.2, 0.6)
    elif reverb_level == "low":
        T60 = uniform(0.1, 0.3)

    return [room_dim, mics, s, T60]

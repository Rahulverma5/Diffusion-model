import os
import numpy as np
import pandas as pd

import warnings
import tensorflow as tf
from configs.config import alphas_cumprod
warnings.filterwarnings('ignore')

def forward_diffusion(x_0, t, noise=None):
    if noise is None:
        noise = tf.random.normal(size=tf.shape(x_0))
    
    t = tf.convert_to_tensor(t, dtype=tf.int32) # Ensure t is a tensor
    mean = tf.gather(tf.sqrt(alphas_cumprod), t)[:, None, None, None] * x_0
    std = tf.gather(tf.sqrt(1 - alphas_cumprod), t)[:, None, None, None]
    return mean + std * noise, noise


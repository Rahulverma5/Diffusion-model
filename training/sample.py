import os
import numpy as np
import pandas as pd

import warnings
import tensorflow as tf
from configs.config import IMG_SIZE, alphas, betas, alphas_cumprod

warnings.filterwarnings('ignore')

def sample(model, T, num_samples=10):
    """This function is used to sample from the diffusion model."""
    
    x_T = tf.random.normal((num_samples, IMG_SIZE, IMG_SIZE, 1))

    for t in reversed(range(T)):
        noise_pred = model(x_T, training=False)
        alpha_t = alphas[t]
        alphas_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        x_T  = 1/np.sqrt(alpha_t) * (x_T - (beta_t / np.sqrt(1 - alphas_cumprod_t)) * noise_pred) 

        if t > 0:
            noise = np.random.normal(size=x_T.shape)
            x_T += np.sqrt(beta_t) * noise
        
        return (x_T + 1) / 2
     
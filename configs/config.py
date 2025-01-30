import numpy as np
import pandas as pd

IMG_SIZE = 28
BATCH_SIZE = 64
T = 1000  # Number of timesteps

beta_start, beta_end = 1e-4, 0.02
betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)

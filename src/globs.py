from typing import Optional
import numpy as np
import random

rnd_seed = -1
rng: Optional[np.random.Generator] = None  # Explicit type hint


def compute_rng(seed_is_random=True, rnd_seed_=1052199760) -> np.random.Generator:
    global rnd_seed, rng
    if rnd_seed == -1:
        if seed_is_random:
            rnd_seed = int(np.random.rand() * (2 ** 32 - 1))
            print(f"New (true) random seed: {rnd_seed}")
        else:
            rnd_seed = rnd_seed_
            print(f"Fixed random seed: {rnd_seed}")
    np.random.seed(rnd_seed)  # for legacy np.random
    rng_ = np.random.default_rng(rnd_seed)
    random.seed(rnd_seed)
    return rng_, rnd_seed


mic0_idx = 0
mod0_idx = 0
if not (mic0_idx == 0 and mod0_idx == 0):
    raise NotImplementedError("Check code before changing mic0_idx or mod0_idx. The variables are not used everywhere.")

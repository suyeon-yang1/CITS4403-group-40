import random
import numpy as np

# Reproducibility
RANDOM_SEED = 42
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)

set_seed(RANDOM_SEED)
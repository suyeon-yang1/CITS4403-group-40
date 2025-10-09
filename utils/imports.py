import sys, os
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.ode_model import ODEParams, lotka_volterra
from src.abm_model import ABMParams, run_abm
from utils.helper_functions import (
    replicate_abm,
    mean_ci,
    plot_with_ci,
    extinction_prob
)
from utils.seed import set_seed



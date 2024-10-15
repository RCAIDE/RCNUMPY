# __init__.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import numpy as np
import scipy as sp
try:
    import jax as jax
except:
    jax = None

# Set the default environment
use_jax      = False
jax_handle   = jax
numpy_handle = np
scipy_handle = sp

from .src import *
from .linalg import *
from .lax import *
from .scipy import *


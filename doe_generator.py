import numpy as np
from smt.design_space import *
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
import pandas as pd

# Create Design Space

variables = ['mdot', 'DH_mid', 'incidence', 'Vexp', 'lean_compound', 'lean_straight', 'tip_clearance', 'n_blade']

ds = DesignSpace([
    FloatVariable(0.025,0.075),     # mdot
    FloatVariable(0.75,0.9),        # DH_mid
    FloatVariable(-10,5),           # incidence
    FloatVariable(-2,0),            # Vexp
    FloatVariable(0,10),            # compound max lean (degrees)
    FloatVariable(0,90),            # straight lean (degrees)    
    FloatVariable(1,5),             # tip_clearance (%)
    IntegerVariable(5,9),           # n_blade
])

test_variables = ['mdot', 'DH_mid']

test_ds = DesignSpace([
    FloatVariable(0.025,0.075),     # mdot
    FloatVariable(0.75,0.9),        # DH_mid
])

# Create sample object

n_samples = 80
ds.seed = 42
sampler = MixedIntegerSamplingMethod(LHS, 
                                    ds, 
                                    criterion = 'ese',
                                    seed = 42)

# Return DoE parameters
x_sampled = sampler(n_samples)

# Save params to csv
data = pd.DataFrame(x_sampled, columns = variables)
file = 'doe_params.csv'
data.to_csv(file)

print(data.head())


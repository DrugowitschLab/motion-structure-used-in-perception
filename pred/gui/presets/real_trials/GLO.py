import numpy as np

human_readable_dsl = "glo"

# MOTION STRUCTURE COMPONENT MATRIX
# each row describes one dot (N dots)
# each column is a motion source (M sources)
B = np.array([
    [1, +1, 0, 1,0,0,0,0,0,0],
    [1, +1, 0, 0,1,0,0,0,0,0],
    [1, +1, 0, 0,0,1,0,0,0,0],
    [1, -1, 0, 0,0,0,1,0,0,0],
    [1, -1, 0, 0,0,0,0,1,0,0],
    [1, -1, 0, 0,0,0,0,0,1,0],
    [1,  0, 0, 0,0,0,0,0,0,1],
    ], dtype=np.float64)

volatility_factor = 8.0             # Makes the stimulus change more rapidly without affecting the covariance matrix
speed_factor = 1.5
glo = 3/3

# MOTION STRENGTHS
# strength of the components (columns of B)
lam_tot = 1/2
lam_I = 1/12
lam_G = np.sqrt(glo) * np.sqrt(lam_tot**2 - lam_I**2)
lam_C = np.sqrt(lam_tot**2 - lam_G**2 - lam_I**2)
lam_M = np.sqrt(lam_tot**2 - lam_G**2)

lam = np.sqrt(volatility_factor) * np.array([lam_G, 0., 0.] + [lam_I]*7 + [lam_M]*0)
lam *= speed_factor

# THE FULL MOTION STRUCTURE MATRIX WILL BE: L = B @ diag(lam)
# TIME CONSTANT for significant changes in angular velocities (in seconds)
tau_vphi = 8. / volatility_factor

# Target dots to fade out
targets = [5, 6]
disc_color = ( np.array([0,0,0,2,2,3,5], dtype='float') + 0.5 ) / 12

# load /mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/alpha_in_grads_0.csv and plot
import matplotlib.pyplot as plt
import numpy as np

alpha_in_grads_0 = np.loadtxt(
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/alpha_in_grads_0.csv", delimiter=","
)
alphas = alpha_in_grads_0.T[0]
in_grads = alpha_in_grads_0.T[1:].T

plt.plot(alphas, in_grads)
plt.savefig("test_0.png")
plt.show()

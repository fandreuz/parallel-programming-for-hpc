import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("solution.dat")[:,2]
data = data.reshape(int(np.sqrt(len(data))), -1)

plt.figure(figsize=(8, 6))
plt.pcolormesh(data)
plt.colorbar()

plt.show()
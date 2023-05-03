import re
import numpy as np
import matplotlib.pyplot as plt
from string import Template

dirs = ("diffusion_fft", "fft")
file_patterns = (Template("fftw_${n}.out"), Template("my_${n}.out"))
procs = np.array((1, 2, 4, 8))

data = {dir: {} for dir in dirs}
for dir, file_pattern in zip(dirs, file_patterns):
    for proc in procs:
        times = []
        with open(f"{dir}/" + file_pattern.substitute(n=proc), "r") as file:
            for line in file.readlines():
                matcher = re.search("time per iteration (\d+\.\d+)", line)
                if matcher:
                    times.append(float(matcher.group(1)))
        data[dir][proc] = np.array(times).mean()


def as_list(dc):
    return [dc[proc] for proc in procs]


plt.figure(figsize=(20, 6))

width = 0.25
plt.bar(procs, as_list(data["diffusion_fft"]), width, edgecolor="k", label="FFTW")
plt.bar(procs + width, as_list(data["fft"]), width, edgecolor="k", label="MPI_Alltoallv")

plt.ylabel("Seconds/iteration")
plt.xlabel("MPI processes")

plt.title("Grid = $512 \\times 512 \\times 512$ (averaged over 100 time steps)")
plt.legend()

plt.yscale("log")

plt.show()

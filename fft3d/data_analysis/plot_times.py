import re
import numpy as np
import matplotlib.pyplot as plt
from string import Template

dirs = ("diffusion_fft", "fft", "4/diffusion_fft", "4/fft")
file_patterns = (
    Template("fftw_${n}.out"),
    Template("my2_${n}.out"),
    Template("fftw_${n}.out"),
    Template("my2_${n}.out"),
)
procs = np.array((1, 2, 4, 8))
labels = ["FFTW", "MPI_Alltoallv"]

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

width = 0.2

dfft1 = plt.bar(
    procs - width * 3 / 2,
    as_list(data["diffusion_fft"]),
    width,
    edgecolor="k",
    label="FFTW",
)
fft1 = plt.bar(
    procs - width / 2,
    as_list(data["fft"]),
    width,
    edgecolor="k",
    label="FFTW",
)
dfft4 = plt.bar(
    procs + width / 2,
    as_list(data["4/diffusion_fft"]),
    width,
    edgecolor="k",
    label="FFTW",
)
fft4 = plt.bar(
    procs + width * 3 / 2,
    as_list(data["4/fft"]),
    width,
    edgecolor="k",
    label="MPI_Alltoallv",
)

(dummy1,) = plt.plot([0], marker="None", linestyle="None", label="dummy-tophead")
(dummy4,) = plt.plot([0], marker="None", linestyle="None", label="dummy-empty")

plt.ylabel("Seconds/iteration")
plt.xlabel("MPI processes")

plt.xticks([1, 2, 4, 8], [1, 2, 4, 8])

plt.title("Grid = $512 \\times 512 \\times 512$ (averaged over 100 time steps)")
plt.legend(
    [dummy1, dfft1, fft1, dummy4, dfft4, fft4],
    [r"$\bf{1 MPI/node}$"] + labels + [r"$\bf{4 MPI/node}$"] + labels,
    ncol=2,
)

plt.show()

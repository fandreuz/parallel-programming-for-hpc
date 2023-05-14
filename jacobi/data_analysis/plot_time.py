import re
import numpy as np
import matplotlib.pyplot as plt

dirs = ("mpi", "acc")
procs = np.array((1, 2, 4, 8))

data = {dir: {} for dir in dirs}

for dir in dirs:
    for proc in procs:
        times = []
        with open(f"{dir}/jacobi_{dir}_{proc}.out", "r") as file:
            for line in file.readlines():
                matcher = re.search("(\d+\.\d+) seconds", line)
                if matcher:
                    times.append(float(matcher.group(1)))
        data[dir][proc] = np.array(times).mean()


def as_list(dc):
    return [dc[proc] for proc in procs]


plt.figure(figsize=(20, 6))

width = 0.25
plt.bar(procs, as_list(data["mpi"]), width, edgecolor="k", label="MPI")
plt.bar(
    procs + width, as_list(data["acc"]), width, edgecolor="k", label="OpenACC + MPI"
)

plt.ylabel("Seconds")
plt.xlabel("MPI processes")

plt.title("$10000 \\times 10000$ (1000 iterations)")
plt.legend()

plt.yscale("log")

plt.show()

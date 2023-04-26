import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import argparse

parser = argparse.ArgumentParser(prog="DataAnalysis")
parser.add_argument("-s", "--size", type=int)
args = parser.parse_args()

mpi_procs = np.array([1, 2, 4, 8])
sizes = np.array([1000, 2500, 5000])
modes = np.array([0, 2, 3])

data = {
    (proc, size, mode): np.loadtxt(f"dir{proc}_{size}_{mode}/proc0.out")
    for proc, size, mode in product(mpi_procs, sizes, modes)
}


def process(d):
    return d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean(), d[:, 3].mean()


def process_pair(mode, size):
    comm_preps = []
    comms = []
    comps = []
    gpus = []
    for proc in mpi_procs:
        comm_prep, comm, comp, gpu = process(data[(proc, size, mode)])
        comm_preps.append(comm_prep)
        comms.append(comm)
        comps.append(comp)
        gpus.append(gpu)
    comm_preps = np.array(comm_preps)
    comms = np.array(comms)
    comps = np.array(comps)
    gpus = np.array(gpus)

    return comm_preps, comms, comps, gpus


size = args.size

plt.figure(figsize=(20, 6))

comm_preps_0, comm_0, comp_0, gpu_0 = process_pair(0, size)
comm_preps_2, comm_2, comp_2, gpu_2 = process_pair(2, size)
comm_preps_3, comm_3, comp_3, gpu_3 = process_pair(3, size)

width = 0.25

comm_line_0 = plt.bar(mpi_procs, comm_0, width, color='r', edgecolor='k')
comp_line_0 = plt.bar(mpi_procs, comp_0, width, bottom=comm_0, color='b', edgecolor='k')

plt.bar(mpi_procs + width, comm_2, width, color='r', edgecolor='k')
comp_line_2 = plt.bar(mpi_procs + width, comp_2, width, bottom=comm_2, color='b', edgecolor='k')

plt.bar(mpi_procs + 2 * width, comm_3, width, color='r', edgecolor='k')
gpu_line_3 = plt.bar(
    mpi_procs + 2 * width,
    gpu_3,
    width,
    bottom=comm_3,
    color='g',
    edgecolor='k'
)
comp_line_3 = plt.bar(
    mpi_procs + 2 * width,
    comp_3,
    width,
    bottom=comm_3 + gpu_3,
    color='b',
    edgecolor='k'
)

plt.xticks(mpi_procs, mpi_procs)
plt.xlabel("MPI processes (1 per node)")
plt.ylabel("Seconds")

def add_label(bars, label):
	target = bars[0]
	plt.text(target.xy[0], target.xy[1] + 1.1 * target._height, label, fontsize=10)
add_label(comp_line_0, "Naive")
add_label(comp_line_2, "OpenBLAS")
add_label(comp_line_3, "cuBLAS")

plt.yscale("log")

plt.title(f"{size} x {size} matmul")

plt.legend((comm_line_0, gpu_line_3, comp_line_0), ("Communication", "Host-device transfer", "Computation"))

plt.show()

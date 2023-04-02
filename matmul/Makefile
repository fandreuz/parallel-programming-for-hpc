.DEFAULT_GOAL := compile

SIZE = 100
MODE = 0

FLAGS = -Wall -Wextra
HPC_FLAGS = -O3
COMPILER = mpic++
UTILITIES = identityMatrix.cpp utils.cpp
SRC = matmulSquareDistributed.cpp

ifeq "$(MODE)" "2"
OPENBLAS_FLAGS = -I ${OPENBLAS_HOME}/include/ -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
else
OPENBLAS_FLAGS =
endif

compile:
	$(COMPILER) $(FLAGS) $(OPENBLAS_FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} -D MODE=${MODE} $(UTILITIES) $(SRC)
output:
	$(COMPILER) $(FLAGS) $(OPENBLAS_FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} -D MODE=${MODE} -D OUTPUT $(UTILITIES) $(SRC)
gpu:
	nvcc -O3 -D SIZE=${SIZE} -D MODE=3 identityMatrix.cpp utils.cpp -x cu matmulSquare.cpp -I /cineca/prod/opt/compilers/spectrum_mpi/10.4.0/binary/include -L /cineca/prod/opt/compilers/spectrum_mpi/10.4.0/binary/lib -lmpi_ibm -lcublas
gpu_output:
	nvcc -O3 -D SIZE=${SIZE} -D MODE=3 -D OUTPUT identityMatrix.cpp utils.cpp -x cu matmulSquare.cpp -I /cineca/prod/opt/compilers/spectrum_mpi/10.4.0/binary/include -L /cineca/prod/opt/compilers/spectrum_mpi/10.4.0/binary/lib -lmpi_ibm -lcublas
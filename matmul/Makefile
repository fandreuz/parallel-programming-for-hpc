.DEFAULT_GOAL := compile

SIZE = 100
MODE = 0

FLAGS = -Wall -Wextra
HPC_FLAGS = -O3
COMPILER = mpic++
UTILITIES = identityMatrix.cpp utils.cpp
SRC = matmulSquare.cpp

ifeq "$(MODE)" "2"
OPENBLAS_FLAGS = -I ${OPENBLAS_HOME}/include/ -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
else
OPENBLAS_FLAGS =
endif

compile:
	$(COMPILER) $(FLAGS) $(OPENBLAS_FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} -D MODE=${MODE} $(UTILITIES) $(SRC)
output:
	$(COMPILER) $(FLAGS) $(OPENBLAS_FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} -D MODE=${MODE} -D OUTPUT $(UTILITIES) $(SRC)
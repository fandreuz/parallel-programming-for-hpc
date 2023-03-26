.DEFAULT_GOAL := compile

FLAGS = -std=c++17 -Wall -Wextra
HPC_FLAGS = -O3 -march=native
COMPILER = mpic++
UTILITIES = identityMatrix.cpp utils.cpp
SRC = matmulSquare.cpp

SIZE = 100

compile:
	$(COMPILER) $(FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} $(UTILITIES) $(SRC)
output:
	$(COMPILER) $(FLAGS) $(HPC_FLAGS) -D SIZE=${SIZE} -D OUTPUT $(UTILITIES) $(SRC)
CC=mpicc
CFLAGS = -O3 -Wall -I$(FFTW_INC)
LIBS = -L$(FFTW_LIB) -lfftw3 -lm
EXE = diffusion.x
SRC = fft_wrapper.c plot_data.c derivative.c diffusion.c
HEAD = utilities.h
OBJ = $(SRC:.c=.o)

compile:
    	$(CC) $(CFLAGS) $(LIBS) $(SRC) -o $(EXE) -std=c11 -pedantic
CC=mpicc
CFLAGS = -O3 -Wall -I$(FFTW_INC)
LIBS = -L$(FFTW_LIB) -lfftw3 -lm
EXE = diffusion.x
SRC = diffusion.c plot_data.c fft_wrapper.c derivative.c 
HEAD = utilities.h
OBJ = $(SRC:.c=.o)

compile:
	$(CC) $(CFLAGS) $(LIBS) $(SRC) -o $(EXE)
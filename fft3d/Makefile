CC=mpicc
CFLAGS = -O3 -g -Wall -I./ -I$(FFTW_INC) -I/u/shared/programs/x86_64/openmpi/1.8.3/gnu/4.9.2/torque/include/ 
LIBS = -L$(FFTW_LIB) -lfftw3_mpi -lfftw3 -lm		
EXE = diffusion.x
SRC = ../diffusion_fft/diffusion.c ../diffusion_fft/plot_data.c ../diffusion_fft/derivative.c fft_wrapper.c fft_3d.c
HEAD = utilities.h
OBJ = $(SRC:.c=.o)


all: $(EXE)

$(EXE): $(OBJ)
	$(CC) *.o $(LIBS) -o $@

%.o : %.c $(HEAD)
	$(CC) $(CFLAGS) -include $(HEAD) -c $<

flush:
	rm -f *.dat

clean: 
	rm -f *.o *.x *~

.DEFAULT_GOAL := compile

compile:
	mpicc -O3 jacobi.c
acc:
	mpicc -O3 -acc -ta=tesla -fast -Minfo=all --diag_suppress implicit_func_decl jacobi.c

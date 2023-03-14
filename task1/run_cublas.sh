pgcc -Mcudalib=cublas -acc -Minfo=accel -o par_dbl main_cublas.c 
pgcc -Mcudalib=cublas -acc -Minfo=accel -D FLOAT -o par_flt main_cublas.c
nvprof ./par_dbl
nvprof ./par_flt
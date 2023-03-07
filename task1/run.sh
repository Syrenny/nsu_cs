pgcc -ta=host -Minfo=accel -o cpu1_dbl task1.c
pgcc -ta=host -Minfo=accel -D FLOAT -o cpu1_flt task1.c
pgcc -ta=multicore -Minfo=accel -o cpum_dbl task1.c
pgcc -ta=multicore -Minfo=accel -D FLOAT -o cpum_flt task1.c
pgcc -acc -Minfo=accel -o par_dbl task1.c 
pgcc -acc -Minfo=accel -D FLOAT -o par_flt task1.c
time ./cpu1_dbl
time ./cpu1_flt
time ./cpum_dbl
time ./cpum_flt
nvprof ./par_dbl
nvprof ./par_flt

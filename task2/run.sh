echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
echo CPU multicore computations
pgc++ -ta=multicore -o main main.cpp 
./main -size 128
./main -size 256
./main -size 512
./main -size 1024
echo +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
echo CPU host computations 
pgc++ -ta=host -o main main.cpp
./main -size 128
./main -size 256
./main -size 512
echo +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
echo GPU computations
pgc++ -acc -o main main.cpp
nsys profile --trace=openacc -o gpu_128 main -size 128 -max_iter 100
nsys profile --trace=openacc -o gpu_256 main -size 256 -max_iter 100
nsys profile --trace=openacc -o gpu_512 main -size 512 -max_iter 100
nsys profile --trace=openacc -o gpu_1024 main -size 1024 -max_iter 100

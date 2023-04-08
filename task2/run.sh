rm reports.txt
touch reports.txt
echo CPU multicore computations >> reports.txt
echo CPU multicore computations
echo ... 
pgc++ -ta=multicore -o main main.cpp 
./main -size 128 >> reports.txt
./main -size 256 >> reports.txt
./main -size 512 >> reports.txt
echo done.
echo CPU host computations >> reports.txt
echo CPU host computations
echo ...
pgc++ -ta=host -o main main.cpp
./main -size 128 >> reports.txt
./main -size 256 >> reports.txt
./main -size 512 >> reports.txt
echo done.
rm *.nsys-rep
echo GPU computations >> reports.txt
echo GPU computations
echo ...
pgc++ -acc -o main main.cpp
./main -size 128 -max_iter 1000000 >> reports.txt
./main -size 256 -max_iter 1000000 >> reports.txt
./main -size 512 -max_iter 1000000 >> reports.txt
./main -size 1024 -max_iter 1000000 >> reports.txt
nsys profile --trace=openacc,nvtx -o gpu_128 main -size 128 -max_iter 50 >> reports.txt
nsys profile --trace=openacc,nvtx -o gpu_256 main -size 256 -max_iter 50 >> reports.txt
nsys profile --trace=openacc,nvtx -o gpu_512 main -size 512 -max_iter 50 >> reports.txt
nsys profile --trace=openacc,nvtx -o gpu_1024 main -size 1024 -max_iter 50 >> reports.txt
echo done.
echo end.
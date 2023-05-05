rm reports.txt
touch reports.txt
rm *.nsys-rep
echo GPU computations >> reports.txt
echo GPU computations
echo ...
nvc++ main.cu -o main -lcudart 
./main -size 128 >> reports.txt
./main -size 256 >> reports.txt
./main -size 512 >> reports.txt
./main -size 1024 >> reports.txt
nsys profile --trace=cuda -o gpu_128 main -size 128 -max_iter 50 >> reports.txt
nsys profile --trace=cuda -o gpu_256 main -size 256 -max_iter 50 >> reports.txt
nsys profile --trace=cuda -o gpu_512 main -size 512 -max_iter 50 >> reports.txt
nsys profile --trace=cuda -o gpu_1024 main -size 1024 -max_iter 50 >> reports.txt
echo done.
echo end.

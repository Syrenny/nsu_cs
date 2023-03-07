echo +++++++++++++++++++++++++++++++++++++++++++++
echo CPU multicore computations
pgc++ -ta=multicore -o main main.cpp 
nsys profile --trace=openacc -o cpu_m_128 main -size 128
nsys profile --trace=openacc -o cpu_m_256 main -size 256
nsys profile --trace=openacc -o cpu_m_512 main -size 512
nsys profile --trace=openacc -o cpu_m_1024 main -size 1024
echo +++++++++++++++++++++++++++++++++++++++++++++
echo CPU host computations 
pgc++ -ta=host -o main main.cpp
nsys profile --trace=openacc -o cpu_128 main -size 128
nsys profile --trace=openacc -o cpu_256 main -size 256
nsys prodile --trace=openacc -o cpu_512 main -size 512
nsys profile --trace=openacc -o cpu_1024 main -size 1024
echo ++++++++++++++++++++++++++++++++++++++++++++++
echo GPU computations
pgc++ -acc -gpu=fastmath -o main main.cpp
nsys profile --trace=openacc -o gpu_128 main -size 128
nsys profile --trace=openacc -o gpu_256 main -size 256
nsys profile --trace=openacc -o gpu_512 main -size 512
nsys profile --trace=openacc -o gpu_1024 main -size 1024

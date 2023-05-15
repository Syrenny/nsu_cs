#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdarg>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// Период вычисления ошибки
#define MAX_THREADS 1024
#define ERROR_COMPUTATION_STEP 100

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
my_cudaFree(7, A, A_new, device_A, device_A_new, device_error, device_error_matrix, temp_stor);	\
exit(-1);	\
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

// Главная функция - расчёт поля 
__global__ void calculate_new_matrix(double* A, double* A_new, size_t size)
{
    unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x;

	// Проверка, чтобы не выходить за границы массива и не пересчитывать граничные условия	
	if(!(i == 0 || j == 0 || i >= size - 1 || j >= size - 1))
	{
		A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}

__global__ void calculate_device_error_matrix(double* A, double* A_new, double* output_matrix, size_t size)
{
	unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;

	size_t idx = i * size + j;
	// Проверка, чтобы не выходить за границы массива и не пересчитывать граничные условия	
	if(!(j == 0 || i == 0 || j >= size - 1 || i >= size - 1))
	{
		output_matrix[idx] = std::abs(A_new[idx] - A[idx]);
	}
}

// Функция для вывода содержимого матрицы 
void print_matrix(double* mx, int n, int m)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout << mx[n * i + j] << " ";
        }
        std::cout << std::endl;
    }
}

void my_cudaFree(int num, double* set...)
{
	va_list args;
	va_start(args, set);
	for(int i = 0; i < num; i++)
	{
		double* ptr = va_arg(args, double*);
		if(ptr)
		{
			cudaFree(ptr);
		}
	}
	va_end(args);
}

double corners[4] = { 10, 20, 30, 20 };

int main(int argc, char** argv)
{
	double *A, *A_new, *device_A, *device_A_new, *device_error, *device_error_matrix, *temp_stor = NULL;
	// Получаем значения из командной строки
	int size = 128, max_iter = 1000000;
    double min_error = 1e-6;
    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]).find("-size") != std::string::npos)
        {
            size = std::atoi(argv[i + 1]);
			if(size < 0)
			{
				std::cout << "Invalid -size parameter" << std::endl;
				return -1;
			}
        }
        else if (std::string(argv[i]).find("-max_iter") != std::string::npos)
        {
            max_iter = std::atoi(argv[i + 1]);
			if(max_iter < 0)
			{
				std::cout << "Invalid -max_iter parameter" << std::endl;
				return -1;
			}
        }
        else if (std::string(argv[i]).find("-min_error") != std::string::npos)
        {
            min_error = std::stod(argv[i + 1]);
			if(min_error <= 0)
			{
				std::cout << "Invalid -min_error parameter" << std::endl;
				return -1;
			}
        }
    }
    std::cout << "Size = " << size << std::endl;
    // Выделение места сразу в pinned памяти 
	size_t full_size = size * size;
	CUDA_CHECK_ERROR(cudaMallocHost(&A_new, full_size * sizeof(double)));
	CUDA_CHECK_ERROR(cudaMallocHost(&A, full_size * sizeof(double)));
	
	std::memset(A, 0, full_size * sizeof(double));

	// Заполнение граничных условий
	A[0] = corners[0];
    A[size - 1] = corners[1];
    A[size * size - 1] = corners[2];
    A[size * (size - 1)] = corners[3];
    double step = (corners[1] - corners[0]) / (size - 1);

	for (int i = 1; i < size - 1; i++) 
    {
        A[i] = corners[0] + i * step;
        A[i * size + (size - 1)] = corners[1] + i * step;
        A[i * size] = corners[0] + i * step;
        A[size * (size - 1) + i] = corners[3] + i * step;
    }

    //print_matrix(A, size, size);
	std::memcpy(A_new, A, full_size * sizeof(double));

	// Выбор устройства
	cudaSetDevice(0);

	// Выделяем память на девайсе и копируем память с хоста на устройство
	// Ошибки обрабатываются в макросе, в случае чего освобождается память
	size_t temp_stor_size = 0;
	CUDA_CHECK_ERROR(cudaMalloc(&device_A, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_A_new, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_error, sizeof(double)));	
	CUDA_CHECK_ERROR(cudaMalloc(&device_error_matrix, sizeof(double) * full_size));	
	CUDA_CHECK_ERROR(cudaMemcpy(device_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice));	
	CUDA_CHECK_ERROR(cudaMemcpy(device_A_new, A_new, sizeof(double) * full_size, cudaMemcpyHostToDevice));			 

	// temp_stor = NULL, функция записывает количество байтов для временного хранилища в temp_stor_size
	cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);	
	// Выделяем память для временного хранилища
	CUDA_CHECK_ERROR(cudaMalloc(&temp_stor, temp_stor_size));

	int iter = 0; 
	double error = 1.0;
	/**
	Количество нитей ограничено 1024, поэтому устанавливаю значение <= 1024,
	а затем вычисляю необходимое количество блоков
	*/
	size_t threads = (size < MAX_THREADS) ? size : MAX_THREADS;
    unsigned int blocks = full_size / threads;
	/*
	dim3 - специальный тип на основе uint3 для задания размерности, 
	имеет удобный конструктор, который инициализирует незаданные компоненты 1.
	Далее инициализирую одномерные векторы blockDim и gridDim.
	*/
	dim3 blockDim(threads);
    dim3 gridDim(blocks);
	// Главный цикл
	clock_t start = clock();
	nvtxRangePushA("Main loop");
	while(iter < max_iter && error > min_error)
	{
        iter += 2;
		// Расчет матрицы
		//<<<размерность сетки, размерность блоков>>>
		// Функция вызывается дважды за итерацию, таким образом избавился от std::swap
        calculate_new_matrix<<<gridDim, blockDim>>>(device_A, device_A_new, size);
		calculate_new_matrix<<<gridDim, blockDim>>>(device_A_new, device_A, size);
		// Расчитываем ошибку с заданным периодом
		if(iter % ERROR_COMPUTATION_STEP == 0)
		{
			// Подсчитываем матрицу ошибок 
			calculate_device_error_matrix<<<gridDim, blockDim>>>(device_A, device_A_new, device_error_matrix, size);
			// Находим максимальное значение ошибки
			cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);
			// Отправка данных обратно на хост
            cudaMemcpy(&error, device_error, sizeof(double), cudaMemcpyDeviceToHost);
  		}
	}
    nvtxRangePop();
	clock_t end = clock();
    std::cout << "Computation time(s): " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl << std::endl;

	// Освобождение памяти 
	my_cudaFree(7, A, A_new, device_A, device_A_new, device_error, device_error_matrix, temp_stor);
	return 0;
}

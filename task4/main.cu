#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
// Период вычисления ошибки
#define ERROR_COMPUTATION_STEP 100

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
delete[]A;	\
delete[]A_new;	\
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

// Главная функция - расчёт поля 
__global__ void calculate_new_matrix(double* A, double* A_new, size_t size)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	// Проверка, чтобы не выходить за границы массива и не пересчитывать граничные условия	
	if(!(blockIdx.x == 0 || threadIdx.x == 0 || blockIdx.x > size - 1 || threadIdx.x > size - 1))
	{
		A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}

// Функция рассчета матрицы ошибок
__global__ void calculate_device_error_matrix(double* A, double* A_new, double* output_matrix, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Проверка, чтобы не выходить за границы массива и не пересчитывать граничные условия
	if(!(blockIdx.x == 0 || threadIdx.x == 0 || blockIdx.x > size - 1 || threadIdx.x > size - 1))
	{
		output_matrix[idx] = std::abs(A_new[idx] - A[idx]);
	}
}
// Функция для вывода содержимого матрицы 
void print_matrix(double* mx, int n, int m)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            std::cout << mx[n * i + j] << " ";
        }
        std::cout << std::endl;
    }
}


double corners[4] = { 10, 20, 30, 20 };

int main(int argc, char** argv)
{
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
    // Выделение памяти на хосте
    auto* A = new double[size * size];
    auto* A_new = new double[size * size];
    int full_size = size * size;
	
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
	cudaSetDevice(3);

	// Выделяем память на девайсе и копируем память с хоста на устройство
	double* device_A, *device_A_new, *device_error, *device_error_matrix, *temp_stor = NULL;
	size_t temp_stor_size = 0;
	CUDA_CHECK_ERROR(cudaMalloc(&device_A, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_A_new, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_error, sizeof(double)));	
	CUDA_CHECK_ERROR(cudaMalloc(&device_error_matrix, sizeof(double) * full_size));	
	CUDA_CHECK_ERROR(cudaMemcpy(device_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice));	
	CUDA_CHECK_ERROR(cudaMemcpy(device_A_new, A_new, sizeof(double) * full_size, cudaMemcpyHostToDevice));			 

	// temp_stor = NULL, функция записывает количество байтов для временного хранилища в temp_stor_size
	cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);	
	// Выделяем память для буфера
	CUDA_CHECK_ERROR(cudaMalloc(&temp_stor, temp_stor_size));

	int iter = 0; 
	double error = 1.0;
	/**
	Количество нитей ограничено 1024, поэтому устанавливаю значение <= 1024,
	а затем вычисляю необходимое количество блоков
	*/
	size_t threads = (size < 1024) ? size : 1024;
    unsigned int blocks = size / threads;
	/*
	dim3 - специальный тип на основе uint3 для задания размерности, 
	имеет удобный конструктор, который инициализирует незаданные компоненты 1.
	Далее инициализирую двумерные векторы blockDim и gridDim. 
	Т.к. ограничение по количеству нитей в блоке 1024, а вектор двумерный, 
	делю на 32 (32^2=1024), можно выбрать число и больше 32, но в таком случае 
	блоки будут использоваться неэффективно. Путем тривиальных вычислений получается, что 
	компоненты gridDim нужно умножать на то же число.
	*/
	dim3 blockDim(threads / 32, threads / 32);
    dim3 gridDim(blocks * 32, blocks * 32);
	// Главный цикл
	clock_t start = clock();
	while(iter < max_iter && error > min_error)
	{
        iter++;
		// Расчет матрицы
		//<<<размерность сетки, размерность блоков>>>
        calculate_new_matrix<<<gridDim, blockDim>>>(device_A, device_A_new, size);
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
	// Обмен указателей
        std::swap(device_A, device_A_new);
	}

	clock_t end = clock();
    std::cout << "Computation time(s): " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl << std::endl;

	// Освобождение памяти 
	CUDA_CHECK_ERROR(cudaFree(device_A));
	CUDA_CHECK_ERROR(cudaFree(device_A_new));
	CUDA_CHECK_ERROR(cudaFree(device_error_matrix));
	CUDA_CHECK_ERROR(cudaFree(temp_stor));
	CUDA_CHECK_ERROR(cudaFree(A));
	CUDA_CHECK_ERROR(cudaFree(A_new));
	delete[]A;
	delete[]A_new;
	return 0;
}

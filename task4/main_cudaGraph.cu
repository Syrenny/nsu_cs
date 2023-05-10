#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdarg>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// Период вычисления ошибки
#define ERROR_COMPUTATION_STEP 100


// __global__ - функция вызывается с CPU или GPU и выполняется на GPU
// Главная функция - расчёт поля 
__global__ void calculate_new_matrix(double* A, double* A_new, size_t size)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	// Проверяем, чтобы не выйти в ходе алгоритма за границы памяти, да и граничные условия пересчитывать не нужно	
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
	// Проверяем, чтобы не выйти в ходе алгоритма за границы памяти, да и граничные условия пересчитывать не нужно
	if(!((blockIdx.x == 0 || threadIdx.x == 0 || blockIdx.x > size - 1 || threadIdx.x > size - 1))
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

void my_cudaFree(int num, double* set...)
{
	va_list args;
	va_start(args, set);
	for(int i = 0; i < num; i++)
	{
		cudaFree(va_arg(args, double*));
	}
	va_end(args);
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
			}
        }
        else if (std::string(argv[i]).find("-min_error") != std::string::npos)
        {
            min_error = std::stod(argv[i + 1]);
			if(min_error <= 0)
			{
				std::cout << "Invalid -min_error parameter" << std::endl;
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

	// Выделяем память на девайсе и копируем память с хоста
	double* device_A, *device_A_new, *device_error, *device_error_matrix, *temp_stor = NULL;
	size_t temp_stor_size = 0;

	cudaError_t cuda_status = cudaMalloc(&device_A, sizeof(double) * full_size);
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_A allocation error" << std::endl;
		delete[]A;
		delete[]A_new;
		return -1;
	}

	cuda_status = cudaMalloc(&device_A_new, sizeof(double) * full_size);
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_A_new allocation error" << std::endl;
		delete[]A;
		delete[]A_new;
		cudaFree(device_A);
		return -1;
	}

	cuda_status = cudaMalloc(&device_error, sizeof(double));
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_error allocation error" << std::endl;
		delete[]A;
		delete[]A_new;
		my_cudaFree(2, device_A, device_A_new);
		return -1;
	}

	cuda_status = cudaMalloc(&device_error_matrix, sizeof(double) * full_size); 
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_error_matrix allocation error" << std::endl;
		delete[]A;
		delete[]A_new;
		my_cudaFree(3, device_error, device_A, device_A_new);
		return -1;
	}

	cuda_status = cudaMemcpy(device_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_A copying error" << std::endl;
		delete[]A;
		delete[]A_new;
		my_cudaFree(4, device_error_matrix, device_error, device_A, device_A_new);
		return -1;
	}

	cuda_status = cudaMemcpy(device_A_new, A_new, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	if(cuda_status != cudaSuccess)
	{
		std::cout << "device_A_new copying error" << std::endl;
		delete[]A;
		delete[]A_new;
		my_cudaFree(4, device_error_matrix, device_error, device_A, device_A_new);
		return -1;
	}
	

	// Функция проверяет temp_stor==NULL, получаем размер временного буфера для редукции
	cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);
	
	// Выделяем память для буфера
	cuda_status = cudaMalloc(&temp_stor, temp_stor_size);
    if (cuda_status != cudaSuccess)
    {
        std::cout << "temp_stor allocation error" << std::endl;
		delete[]A;
		delete[]A_new;
		my_cudaFree(4, device_error_matrix, device_error, device_A, device_A_new);
		return -1;
    }
	int iter = 0; 
	double* error;
	cudaMallocHost(&error, sizeof(double));
	*error = 1.0;

	bool isGraphCreated = false;
	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;

	size_t threads = (size < 1024) ? size : 1024;
    unsigned int blocks = size / threads;

	dim3 blockDim(threads / 32, threads / 32);
    dim3 gridDim(blocks * 32, blocks * 32);

	// Главный алгоритм 
	clock_t start = clock();
	nvtxRangePushA("Main loop");
	while(iter < max_iter && *error > min_error)
	{
		// Расчет матрицы
		if (isGraphCreated)
		{
			cudaGraphLaunch(instance, stream);
			
			cudaMemcpyAsync(error, device_error, sizeof(double), cudaMemcpyDeviceToHost, stream);

			cudaStreamSynchronize(stream);

			iter += ERROR_COMPUTATION_STEP;
		}
		else
		{
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
			for(size_t i = 0; i < ERROR_COMPUTATION_STEP / 2; i++)
			{
				calculate_new_matrix<<<gridDim, blockDim, 0, stream>>>(device_A, device_A_new, size);
				calculate_new_matrix<<<gridDim, blockDim, 0, stream>>>(device_A_new, device_A, size);
			}
			// Расчитываем ошибку каждую сотую итерацию
			calculate_device_error_matrix<<<threads * blocks * blocks, threads,  0, stream>>>(device_A, device_A_new, device_error_matrix, size);
			cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size, stream);
	
			cudaStreamEndCapture(stream, &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			isGraphCreated = true;
  		}
	}
    nvtxRangePop();
	clock_t end = clock();
    std::cout << "Computation time(s): " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl << std::endl;

	// Освобождение памяти 
	cudaFree(device_A);
	cudaFree(device_A_new);
	cudaFree(device_error_matrix);
	cudaFree(temp_stor);
	cudaFree(A);
	cudaFree(A_new);
	delete[]A;
	delete[]A_new;
	return 0;
}

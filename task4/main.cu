#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
// Период вычисления ошибки
#define ERROR_COMPUTATION_STEP 100


// Главная функция - расчёт поля 
__global__
void calculate_new_matrix(double* A, double* A_new, size_t size)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	
	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
		A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}

// Функция рассчета матрицы ошибок
__global__
void calculate_device_error_matrix(double* A, double* A_new, double* output_matrix)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(!(blockIdx.x == 0 || threadIdx.x == 0))
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
	// Получаем значения из коммандной строки
	int size = 128, max_iter = 1000000;
    double min_error = 1e-6;
    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]).find("-size") != std::string::npos)
        {
            size = std::atoi(argv[i + 1]);
        }
        else if (std::string(argv[i]).find("-max_iter") != std::string::npos)
        {
            max_iter = std::atoi(argv[i + 1]);
        }
        else if (std::string(argv[i]).find("-min_error") != std::string::npos)
        {
            min_error = std::stod(argv[i + 1]);
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

	cudaError_t cuda_status_1 = cudaMalloc(&device_A, sizeof(double) * full_size);
	cudaError_t cuda_status_2 = cudaMalloc(&device_A_new, sizeof(double) * full_size);
	cudaError_t cuda_status_3 = cudaMalloc(&device_error, sizeof(double));
	cudaError_t cuda_status_4 = cuda_status_1 = cudaMalloc(&device_error_matrix, sizeof(double) * full_size);
	// Проверка статуса выполнения функций 
	if (cuda_status_1 || cuda_status_2 || cuda_status_3 || cuda_status_4 != cudaSuccess)
	{
		std::cout << "Memory allocation error" << std::endl;
		return -1;
	}
	// Копируем области памяти из хоста на устройство
	cuda_status_1 = cudaMemcpy(device_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	cuda_status_2 = cudaMemcpy(device_A_new, A_new, sizeof(double) * full_size, cudaMemcpyHostToDevice);

	if (cuda_status_1 || cuda_status_2 != cudaSuccess)
	{
		std::cout << "Memory transfering error" << std::endl;
		return -1;
	}

	// Получаем размер временного буфера для редукции
	cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);
	
	// Выделяем память для буфера
	cuda_status_1 = cudaMalloc(&temp_stor, temp_stor_size);
    if (cuda_status_1 != cudaSuccess)
    {
        std::cout << "Temporary storage allocation error " << std::endl;
        return -1;
    }
	int iter = 0; 
	double error = 1.0;

	// Главный цикл
	clock_t start = clock();
	while(iter < max_iter && error > min_error)
	{
        iter++;
		// Расчет матрицы
		//<<<количество блоков, количество нитей в блоке>>>
        calculate_new_matrix<<<size - 1, size - 1>>>(device_A, device_A_new, size);
		// Расчитываем ошибку с заданным периодом
		if(iter % ERROR_COMPUTATION_STEP == 0)
		{
			// Подсчитываем матрицу ошибок 
			calculate_device_error_matrix<<<size - 1, size - 1>>>(device_A, device_A_new, device_error_matrix);
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
	cudaFree(device_A);
	cudaFree(device_A_new);
	cudaFree(device_error_matrix);
	cudaFree(temp_stor);
	cudaFree(A);
	cudaFree(A_new);
	return 0;
}

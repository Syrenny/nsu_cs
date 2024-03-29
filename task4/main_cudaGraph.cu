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
my_cudaFree(8, error, A, A_new, device_A, device_A_new, device_error, device_error_matrix, temp_stor);	\
exit(-1);	\
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

// __global__ - функция вызывается с CPU или GPU и выполняется на GPU
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

// Функция рассчета матрицы ошибок
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
	double *error, *A, *A_new, *device_A, *device_A_new, *device_error, *device_error_matrix, *temp_stor = NULL;
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
				exit(-1);
			}
        }
        else if (std::string(argv[i]).find("-max_iter") != std::string::npos)
        {
            max_iter = std::atoi(argv[i + 1]);
			if(max_iter < 0)
			{
				std::cout << "Invalid -max_iter parameter" << std::endl;
				exit(-1);
			}
        }
        else if (std::string(argv[i]).find("-min_error") != std::string::npos)
        {
            min_error = std::stod(argv[i + 1]);
			if(min_error <= 0)
			{
				std::cout << "Invalid -min_error parameter" << std::endl;
				exit(-1);
			}
        }
    }
    std::cout << "Size = " << size << std::endl;
	
    // Выделение памяти на хосте
    int full_size = size * size;
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
	cudaSetDevice(1);

	// Выделяем память на девайсе и копируем память с хоста
	size_t temp_stor_size = 0;

	CUDA_CHECK_ERROR(cudaMalloc(&device_A, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_A_new, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMalloc(&device_error, sizeof(double)));
	CUDA_CHECK_ERROR(cudaMalloc(&device_error_matrix, sizeof(double) * full_size));
	CUDA_CHECK_ERROR(cudaMemcpy(device_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(device_A_new, A_new, sizeof(double) * full_size, cudaMemcpyHostToDevice));	

	// Функция проверяет temp_stor==NULL, получаем размер временного буфера для редукции
	cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size);
	
	// Выделяем память для буфера
	CUDA_CHECK_ERROR(cudaMalloc(&temp_stor, temp_stor_size));

	int iter = 0; 
	CUDA_CHECK_ERROR(cudaMallocHost(&error, sizeof(double)));
	*error = 1.0;

	bool is_graph_inited = false;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	// Cодержит информацию, определяющую структуру и содержание графа
	cudaGraph_t graph;
	// «Исполняемый граф», представление графа в форме, которая может быть запущена и выполнена аналогично одному ядру
	cudaGraphExec_t instance;
	/*
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

	// Главный алгоритм 
	clock_t start = clock();
	nvtxRangePushA("Main loop");
	while(iter < max_iter && *error > min_error)
	{
		// Расчет матрицы
		if (is_graph_inited)
		{
			// Запускаем выполнение графа
			cudaGraphLaunch(instance, stream);
			// Копируем значение ошибки обратно на хост 
			cudaMemcpy(error, device_error, sizeof(double), cudaMemcpyDeviceToHost);
			iter += ERROR_COMPUTATION_STEP;
		}
		else
		{
			// Начало сбора информации о действиях графического процессора, которые отправляются в поток
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
			// 100 раз вычисляем сетку
			for(size_t i = 0; i < ERROR_COMPUTATION_STEP / 2; i++)
			{
				/*
				Расчет матрицы
				<<<размерность сетки, размерность блоков, дополнительный объем shared-памяти, Cuda stream>>>
				Функция вызывается дважды за итерацию, поэтому ERROR_COMPUTATION_STEP / 2
				*/
				calculate_new_matrix<<<gridDim, blockDim, 0, stream>>>(device_A, device_A_new, size);
				calculate_new_matrix<<<gridDim, blockDim, 0, stream>>>(device_A_new, device_A, size);
			}
			// Вычисляем матрицу ошибок
			calculate_device_error_matrix<<<gridDim, blockDim, 0, stream>>>(device_A, device_A_new, device_error_matrix, size);
			cub::DeviceReduce::Max(temp_stor, temp_stor_size, device_error_matrix, device_error, full_size, stream);
			// Конец захвата информации из потока
			cudaStreamEndCapture(stream, &graph);
			// Создает и предварительно инициализирует ядра, чтобы потом их можно было запускать как можно быстрее
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			is_graph_inited = true;
  		}
	}
    nvtxRangePop();
	clock_t end = clock();
    std::cout << "Computation time(s): " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Error: " << *error << std::endl;
    std::cout << "Iteration: " << iter << std::endl << std::endl;

	// Освобождение памяти 
	my_cudaFree(8, error, A, A_new, device_A, device_A_new, device_error, device_error_matrix, temp_stor);
	return 0;
}

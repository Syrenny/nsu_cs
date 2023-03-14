#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef FLOAT 
#define TYPE float
#define FUNC sin
#else 
#define TYPE double
#define FUNC asin
#endif

#define _USE_MATH_DEFINES
//#include <math.h>
#include <omp.h>
#define N 10000000

int main()
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	double start_time, fill_end, sum_end;
	TYPE* my_array = (TYPE*)malloc(sizeof(TYPE) * N);
	TYPE res = 0;
#pragma acc data create(my_array[:N]) copyin(N) copy(res)
	{
		start_time = omp_get_wtime();
#pragma acc parallel loop vector vector_length(160) gang 
		for (int i = 0; i < N; i++)
		{
			TYPE k = (TYPE)i / N;
			my_array[i] = FUNC(k * 2.0 * M_PI);
		}
		fill_end = omp_get_wtime();
		cublasStatus_t stat;
		stat = cublasSasum(handle, N, my_array, 1, res);
		if (stat != CUBLAS_STATUS_SUCCESS)
		{
			print("data download failed");
			free(my_array);
			cublasDestroy(handle);
			return EXIT_FAILURE;
		}
		sum_end = omp_get_wtime();
	}
	free(my_array);
	cublasDestroy(handle);
	printf("to fill: %0.15lf\n", fill_end - start_time);
	printf("to sum: %0.15lf\n", sum_end - fill_end);
	return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define TYPE double
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

void FuncArray(TYPE** my_array, int len)
{
	TYPE* temp = (TYPE*)malloc(sizeof(TYPE) * len);
	for (int i = 0; i < len; i++)
	{
		TYPE k = (TYPE)i / len;
		temp[i] = sin(k * 2.0 * M_PI);
	}
	*my_array = temp;
}

TYPE SumArray(TYPE* my_array, int len)
{
	TYPE res = 0;
	for (int i = 0; i < len; i++)
		res += my_array[i];
	return res;
}

int main()
{
	double start_time, end_time_sin, end_time_sum;
	TYPE* my_array = (TYPE*)malloc(sizeof(TYPE));
	long long len = N;
	start_time = omp_get_wtime();
	FuncArray(&my_array, len);
	end_time_sin = omp_get_wtime();
	/*for (int i = 0; i < len; i++)
		printf("%f ", my_array[i]);*/
		//printf("%f", SumArray(my_array, len));
	SumArray(my_array, len);
	end_time_sum = omp_get_wtime();
	printf("time to compute sin = %g seconds\n", (double)(end_time_sin - start_time));
	printf("time to compute sum = %g seconds\n", (double)(end_time_sum - end_time_sin));
	return 0;
}

#include <stdlib.h>
#include <stdio.h>
#define TYPE double
#ifdef FLOAT 
#define TYPE float
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#define N 10000000

int main()
{
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
			my_array[i] = sin(k * 2.0 * M_PI);
		}
		fill_end = omp_get_wtime();

#pragma acc parallel loop reduction(+:res) vector vector_length(160) gang
		for (int i = 0; i < N; i++)
			res += my_array[i];
		sum_end = omp_get_wtime();
	}
	printf("to fill: %0.15lf\n", fill_end - start_time);
	printf("to sum: %0.15lf\n", sum_end - fill_end);
	return 0;
}

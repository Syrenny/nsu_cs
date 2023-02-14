#include <stdlib.h>
#include <stdio.h>
#define TYPE double
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

int main()
{
	TYPE* my_array = (TYPE*)malloc(sizeof(TYPE)* N);
	TYPE res = 0;
	#pragma acc create(my_array[:N]) copyin(N) copy(res)
	{
		#pragma acc kernels
		for (int i = 0; i < N; i++)
		{
			TYPE k = (TYPE)i / N;
			my_array[i] = sin(k * 2.0 * M_PI);
		}

		#pragma acc kernels reduction(+:res)
		for (int i = 0; i < N; i++)
			res += my_array[i];
	}
	return 0;
}

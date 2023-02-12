#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

void FuncArray(double** my_array, int len)
{
	double* temp = (double*)malloc(sizeof(double) * len);
	#pragma acc parallel 
	{	
		#pragma acc parallel loop
		for (int i = 0; i < len; i++)
		{
			double k = (double)i / len;
			temp[i] = sin(k * 2.0 * M_PI);
		}
	}
	*my_array = temp;
}

double SumArray(double* my_array, int len)
{
	double res = 0;
	#pragma acc parallel
	{
		#pragma acc parallel loop reduction(+:res)
		for (int i = 0; i < len; i++)
			res += my_array[i];
	}
	
	return res;
}

int main()
{
	double* my_array = (double*)malloc(sizeof(double));
	long long len = N;
	FuncArray(&my_array, len);
	/*for (int i = 0; i < len; i++)
		printf("%f ", my_array[i]);*/
		//printf("%f", SumArray(my_array, len));
	SumArray(my_array, len);
	return 0;
}

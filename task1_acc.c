#include <stdlib.h>
#include <stdio.h>
#define TYPE double
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

void FuncArray(TYPE** my_array, int len)
{
	TYPE* temp = (TYPE*)malloc(sizeof(TYPE) * len);
	#pragma acc parallel loop
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
	#pragma acc parallel loop reduction(+:res)
	for (int i = 0; i < len; i++)
		res += my_array[i];
	
	return res;
}

int main()
{
	TYPE* my_array = (TYPE*)malloc(sizeof(TYPE));
	long long len = N;
	FuncArray(&my_array, len);
	/*for (int i = 0; i < len; i++)
		printf("%f ", my_array[i]);*/
		//printf("%f", SumArray(my_array, len));
	SumArray(my_array, len);
	return 0;
}

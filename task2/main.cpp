#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sys/time.h>

struct timeval tv1, tv2, dtv;
struct timezone tz;
void time_start(){ gettimeofday(&tv1, &tz); }
long time_stop()
{
	gettimeofday(&tv2, &tz);
	dtv.tv_sec = tv2.tv_sec - tv1.tv_sec;
	dtv.tv_usec = tv2.tv_usec - tv1.tv_usec;
	if(dtv.tv_usec < 0)
	{
		dtv.tv_sec--;
		dtv.tv_usec += 1000000;
	}
	return dtv.tv_sec * 1000 + dtv.tv_usec / 1000;
}

double corners[4] = { 10, 20, 30, 20 };

int main(int argc, char** argv) 
{
    int size = 128, max_iter = 1000000;
    double min_error = 10e-6;
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
    auto* A = new double[size * size];
    auto* A_new = new double[size * size];

    std::memset(A, 0, sizeof(double) * size * size);

    A[0] = corners[0];
    A[size - 1] = corners[1];
    A[size * size - 1] = corners[2];
    A[size * (size - 1)] = corners[3];

    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);

#pragma acc enter data copyin(A[0:full_size]) create(A_new[0:full_size])
    {
#pragma acc parallel loop seq gang num_gangs(size) vector vector_length(size)
        for (int i = 1; i < size - 1; i++) 
        {
            A[i] = corners[0] + i * step;
            A[i * size + (size - 1)] = corners[1] + i * step;
            A[i * size] = corners[0] + i * step;
            A[size * (size - 1) + i] = corners[3] + i * step;
        }
    }

    std::memcpy(A_new, A, sizeof(double) * full_size);

    double error = 1.0;
    int iter = 0;
    time_start();

#pragma acc enter data copyin(A_new[0:full_size], A[0:full_size], error, iter, min_error, max_iter)
    while (error > min_error && iter < max_iter) 
    {
        iter++;
	error = 0.0;

#pragma acc update device(error)
#pragma acc data present(A, A_new)
#pragma acc parallel loop independent collapse(2) vector vector_length(300) gang num_gangs(300) reduction(max:error) async(1)
        for (int i = 1; i < size - 1; i++)
        {
            for (int j = 1; j < size - 1; j++)
            {
                A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                error = fmax(error, A_new[i * size + j] - A[i * size + j]);
            }
        }
        double* temp = A;
        A = A_new;
        A_new = temp;
    }
#pragma acc update host(error)
    std::cout << "Computation time(ms): " << time_stop() << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    delete[]A;
    delete[]A_new;
    return 0;
}

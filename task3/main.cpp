#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

double corners[4] = { 10, 20, 30, 20 };

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
int main(int argc, char** argv) 
{
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

    auto* A = new double[size * size];
    auto* A_new = new double[size * size];

    std::memset(A, 0, sizeof(double) * size * size);

    A[0] = corners[0];
    A[size - 1] = corners[1];
    A[size * size - 1] = corners[2];
    A[size * (size - 1)] = corners[3];
    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);

    for (int i = 1; i < size - 1; i++) 
    {
        A[i] = corners[0] + i * step;
        A[i * size + (size - 1)] = corners[1] + i * step;
        A[i * size] = corners[0] + i * step;
        A[size * (size - 1) + i] = corners[3] + i * step;
    }
    std::memcpy(A_new, A, sizeof(double) * full_size);

    double error = 1.0;
    cublasHandle_t handle;
	cublasStatus_t stat;
    int iter = 0;
    stat = cublasCreate(&handle);
    clock_t start = clock();
#pragma acc data copyin(A[0:full_size], A_new[0:full_size])
    {
        double alpha = -1.0;
        int idmax = 0;
        while (error > min_error && iter < max_iter) 
        {
            iter++;
#pragma acc data present(A, A_new)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256)
            for (int i = 1; i < size - 1; i++)
            {
                for (int j = 1; j < size - 1; j++)
                {
                    A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                }
            }
            if(iter % 100 == 0)
            {
#pragma acc host_data use_device(A, A_new)
                {
                    stat = cublasDaxpy(handle, full_size, &alpha, A_new, 1, A, 1);
                    stat = cublasIdamax(handle, full_size, A, 1, &idmax);
                }

#pragma acc update host(A[idmax - 1])
                error = std::abs(A[idmax - 1]);

#pragma acc host_data use_device(A, A_new)
                stat = cublasDcopy(handle, full_size, A_new, 1, A, 1);
            }
            double* temp = A;
            A = A_new;
            A_new = temp;
        }
    }
    clock_t end = clock();
    cublasDestroy(handle);
    std::cout << "Computation time(s): " << 1.0 * (end - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl << std::endl;
    delete[]A;
    delete[]A_new;
    return 0;
}

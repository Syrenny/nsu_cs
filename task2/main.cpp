#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>

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

    clock_t start = clock();
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
    start = clock();

#pragma acc enter data copyin(A_new[0:full_size], A[0:full_size], error, iter, min_error, max_iter)
    while (error > min_error && iter < max_iter) 
    {
        iter++;
        error = 0.0;
#pragma acc data present(A, A_new, error)
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

    double end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "Computation time: " << elapsed_secs << std::endl;

#pragma acc data copyout(A[0:full_size], A_new[0:full_size])
#pragma acc update host(A[0:size * size], A_new[0:size * size], error)
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    delete[]A;
    delete[]A_new;
    return 0;
}

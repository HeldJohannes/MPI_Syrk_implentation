//
// Created by Johannes Held on 04.10.23.
//

#ifndef MPI_SYRK_IMPLEMENTATION_MPI_SYRK_IMPLEMENTATION_H
#define MPI_SYRK_IMPLEMENTATION_MPI_SYRK_IMPLEMENTATION_H

typedef struct {
    int m;
    int n;
    int c;
    char *fileName;
} run_config;

/**
 * Usage function.
 * @brief This function writes helpful usage information about the program to stderr.
 * @param myprog the program name
 */
void wrong_usage(char *myProgramName);

void error_exit(int rank, int print_usage, char *name, const char *msg, ...);

void parseInput(int argc, char **argv, int rank);

void printResult(int rank, int len, int cols, int array[]);

void index_calculation(int *arr, int n, int world_size);

void readInputFile(float *input, int rank, char **argv);

void computeInputAndTransposed(int rank, const int *index_arr, const float *input, float *rank_input,
                               float *rank_input_t);

void syrkIterativ(int rank, const int *index_arr, const int rank_input[][config.m],
                  const int rank_input_t[][index_arr[rank]], int rank_result[]);


#endif //MPI_SYRK_IMPLEMENTATION_MPI_SYRK_IMPLEMENTATION_H

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
    char *arg_0;
} run_config;

/**
 * read_input function parses command-line arguments and populates a run_config structure.
 *
 * @param s A pointer to a run_config structure where the parsed values will be stored.
 * @param argc The number of command-line arguments.
 * @param argv An array of strings containing the command-line arguments.
 * @return
 *  EXIT_SUCCESS: if the parsing is successful, otherwise an error code.
 */
int read_input(run_config *s, int argc, char* argv[]);

/**
 * read_input_file function reads the input file and populates the array A with its contents.
 *
 * @param s A pointer to A run_config structure containing the file name.
 * @param A A pointer to a float array where the input values will be stored.
 * @param rank The rank of the current processor
 * @return EXIT_SUCCESS if the file is read successfully, otherwise an error code.
 */
int read_input_file(const int rank, run_config *s, float *A);

/**
 * Prints usage information for the program.
 *
 * @param myProgramName Name of the program
 */
void wrong_usage(char *myProgramName);

/**
 * This function is used to print error messages and exit the program.
 *
 * @param rank Rank of the caller
 * @param print_usage Flag indicating whether to print usage information
 * @param name Name of the program or function where the error occurred
 * @param msg Format string for the error message
 * @param ... Additional arguments for the error message (variable argument list)
 */
void error_exit(int rank, int print_usage, char *name, const char *msg, ...);

void parseInput(run_config *s, int argc, char **argv, int rank);

void printResult(int rank, int len, int cols, float array[]);

void index_calculation(int *arr, int n, int world_size);

void readInputFile(int *input, int rank, char **argv);

void computeInputAndTransposed(run_config *s, int rank, const int *index_arr, const float *input, float *rank_input, float *rank_input_t);

void syrkIterative(run_config *s, int rank, const int *index_arr, const float* rank_input, const float* rank_input_t, float* rank_result);

void transposeMatrix(int m, int n, float *matrix, float *result);

#endif //MPI_SYRK_IMPLEMENTATION_MPI_SYRK_IMPLEMENTATION_H

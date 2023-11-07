#include "log.h"
#include "MPI_Syrk_implementation.h"
#include <limits.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cblas.h>

#define TRUE 1
#define FALSE 0


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

void readInputFile(int *input, int rank, char **argv);

void computeInputAndTransposed(int rank, const int *index_arr, const int *input, int rank_input[][config.m],
                               int rank_input_t[][index_arr[rank]]);

void syrkIterativ(int rank, const int *index_arr, const int rank_input[][config.m],
                  const int rank_input_t[][index_arr[rank]], int rank_result[]);

/**
 *
 *
 *
 * @param argc number of input parameters
 * @param argv  available options are -n and -m; where -m specifies the number of input rows and -n the number of input columns
 * @return 0 if successful
 */
int main(int argc, char *argv[]) {

    // setup:
    log_set_level(LOG_INFO);

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int index_arr[world_size];


    if (argc != 6) {
        error_exit(rank, TRUE, argv[0], "To many or not enough input variables!");
    }

    parseInput(argc, argv, rank);

    //input_matrix_in_array_form
    int input[config.m * config.n];

    readInputFile(input, rank, argv);

    //check the size of col that each node gets
    log_trace("rank = %d, size = %d * m", rank, index_arr[rank]);

    // calculate how many cols each node gets
    index_calculation(index_arr, config.n, world_size);

    //input matrix for each node:
    int rank_input[index_arr[rank]][config.m];

    //transposed input matrix for each node:
    int rank_input_t[config.m][index_arr[rank]];

    // compute the input matrix, and it's transpose,
    // which consists of the columns and all rows in that column:
    computeInputAndTransposed(rank, index_arr, input, rank_input, rank_input_t);

    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode 
    //  and store the result in rank_result
    int rank_result[config.m * config.m];
    for (int i = 0; i < config.m * config.m; ++i) {
        rank_result[i] = 0;
    }
    cblas_ssyrk(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,
            config.m,
            config.n,
            1,
            (const float *) input,
            config.n,
            1,
            (float *) rank_result,
            config.m);

    int counts[world_size];
    index_calculation(counts, config.m * config.m, world_size);

    int reduction_result[counts[rank]];
    for (int i = 0; i < counts[rank]; ++i) {
        reduction_result[i] = 0;
    }

    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }

    if (rank == 0) {
        log_debug("m = %d", config.m);
        int *buffer = (int *) calloc(config.m * config.m, sizeof(int));
        int displacements[world_size];
        displacements[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i - 1] + counts[i - 1];
        }
//        printf("counts:\n");
//        printResult(rank, world_size, counts);
//        printf("displacements:\n");
//        printResult(rank, world_size, displacements);

        MPI_Gatherv(reduction_result, counts[rank], MPI_INT, buffer, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        //Print the result:
        printf("Values gathered in the buffer on process %d:\n", rank);

        printResult(rank, config.m * config.m, config.m, buffer);
        free(buffer);
    } else {
        MPI_Gatherv(reduction_result, counts[rank], MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }


    // finish the program:
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void syrkIterativ(int rank, const int *index_arr, const int rank_input[][config.m],
                  const int rank_input_t[][index_arr[rank]], int rank_result[]) {
    // set all array entries to 0:
    for (int i = 0; i < config.m; ++i) {
        for (int j = 0; j < config.m; ++j) {
            rank_result[i * config.m + j] = 0;
        }
    }
    for (int row = 0; row < config.m; ++row) {
        for (int col = 0; col < config.m; ++col) {
            for (int c = 0; c < index_arr[rank]; ++c) {
                log_trace("index_arr[%d] = %d", rank, index_arr[rank]);
                rank_result[row * config.m + col] =
                        rank_input[c][row] * rank_input_t[col][c] + rank_result[row * config.m + col];
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * config.m + col]);
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * config.m + col]);
            }
        }
    }
}



void computeInputAndTransposed(int rank, const int *index_arr, const int *input, int rank_input[][config.m],
                               int rank_input_t[][index_arr[rank]]) {
    int rank_count = 0;
    for (int i = 0; i < rank; ++i) {
        rank_count += index_arr[i];
        log_trace("rank = %d, rank_count = %d", rank, rank_count);
    }
    for (int col_count = 0; col_count < index_arr[rank]; ++col_count) {
        for (int row_count = 0; row_count < config.m; ++row_count) {
            int tmp = input[(col_count + rank_count) + row_count * config.n];
            rank_input[col_count][row_count] = tmp;
            rank_input_t[row_count][col_count] = tmp;
            log_trace("rank-%d: rank_input[%d][%d] = %d", rank, col_count, row_count, rank_input[col_count][row_count]);
            log_trace("rank-%d: rank_input_t[%d][%d] = %d", rank, row_count, col_count,
                      rank_input_t[row_count][col_count]);
        }
    }
}

void readInputFile(int *input, int rank, char **argv) {
    FILE *stream = fopen(config.fileName, "r");

    if (stream == NULL) {
        error_exit(rank, FALSE, argv[0], "[MPI process %d] Can't open the file: %s", rank, config.fileName);
    }

    char *line = NULL;
    size_t len = 0;
    int counter = 0;
    while (getline(&line, &len, stream) != -1) {
        char *subtoken = strtok(line, ";");
        while (subtoken) {
            char *pEnd;
            long res = strtol(subtoken, &pEnd, 10);
            if (input[counter] > INT_MIN || input[counter] < INT_MAX) {
                input[counter] = (int) res;
            } else {
                log_error("Input is not an integer --> Abort");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            log_trace("[MPI process %d] input[%d] = %d", rank, counter, input[counter]);
            counter++;
            subtoken = strtok(NULL, ";");
        }
    }

    fclose(stream);
}

void wrong_usage(char *myProgramName) {
    log_error("Usage: %s [-m row_count] [-n col_count] [file...]", myProgramName);
}

void index_calculation(int *arr, int n, int world_size) {
    int input_size = n / world_size;

    int rest = n % world_size;
    log_trace("rest = %d", rest);

    for (int i = 0; i < world_size; ++i) {
        arr[i] = input_size;
        if (i < rest) {
            arr[i] += 1;
        }
    }
}

void printResult(int rank, int len, int cols, int array[]) {
    FILE *file = fopen("result.csv", "w");
    //char *string = "[MPI process %d] ";
    //fprintf(file ,string , rank);
    for (int j = 0; j < cols; ++j) {
        for (int i = j * cols; i < j * cols + cols; ++i) {
            if (i == j * cols + cols - 1) {
                fprintf(file, "%d", array[i]);
            } else {
                fprintf(file, "%d; ", array[i]);
            }
        }
        fprintf(file, "\n");
    }
    printf("\n");
}

void parseInput(int argc, char **argv, int rank) {

    //total_row_number
    config.m = -1;
    //total_col_number
    config.n = -1;

    int opt;
    char *end;
    while ((opt = getopt(argc, argv, "m:n:")) != -1) {
        switch (opt) {
            case 'm':
                config.m = (int) strtol(optarg, &end, 10);
                break;
            case 'n':
                config.n = (int) strtol(optarg, &end, 10);
                break;
            default:
            case '?':
                error_exit(rank, TRUE, argv[0], "wrong usage: option %c doesn't exist", opt);
        }
    }

    log_trace("m = %d; n = %d", config.m, config.n);
    if (config.m == -1 || config.n == -1) {
        error_exit(rank, TRUE, argv[0], "parameters m (= %d) and n (= %d) have to be bigger than 0", config.m,
                   config.n);
    }

    log_trace("optind = %d, argc = %d", optind, argc);
    config.fileName = argv[optind];
}

void error_exit(int rank, int print_usage, char *name, const char *msg, ...) {
    if (rank == 0) {

        char buf[16];
        time_t t = time(NULL);
        buf[strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&t))] = '\0';
        fprintf(stderr, "%s %-5s : ", buf, "ERROR");

        va_list ap;
        va_start(ap, msg);
        vfprintf(stderr, msg, ap);
        va_end(ap);
        fprintf(stderr, "\n");
        if (print_usage) {
            wrong_usage(name);
        }
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
}

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>
#include <limits.h>
#include <time.h>
#include "log.h"

#define TRUE 1
#define FALSE 0

/**
 * Usage function.
 * @brief This function writes helpful usage information about the program to stderr.
 * @param myprog the program name
 */
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

void display(int rank, int len, int array[]) {
    char *string = "[MPI process %d] ";
    printf(string, rank);
    for (int i = 0; i < len; ++i) {
        printf(" %d, ", array[i]);
    }
    printf("\n");
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
    log_set_level(LOG_DEBUG);

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int index_arr[world_size];


    if (argc != 6) {
        error_exit(rank, TRUE, argv[0], "To many or not enough input variables!");
    }

    //total_row_number
    int m = -1;

    //total_col_number
    int n = -1;

    int opt;
    char *end;
    while ((opt = getopt(argc, argv, "m:n:")) != -1) {
        switch (opt) {
            case 'm':
                m = (int) strtol(optarg, &end, 10);
                break;
            case 'n':
                n = (int) strtol(optarg, &end, 10);
                break;
            default:
            case '?':
                error_exit(rank, TRUE, argv[0], "wrong usage: option %c doesn't exist", opt);
        }
    }

    log_trace("m = %d; n = %d", m, n);

    if (m == -1 || n == -1) {
        error_exit(rank, TRUE, argv[0], "parameters m (= %d) and n (= %d) have to be bigger than 0", m, n);
    }

    //input_matrix_in_array_form
    int input[m * n];

    log_trace("optind = %d, argc = %d", optind, argc);

    char *fileName = argv[optind];

    FILE *stream = fopen(fileName, "r");

    if (stream == NULL) {
        error_exit(rank, FALSE, argv[0], "[MPI process %d] Can't open the file: %s", rank, fileName);
    }

    char line[100];
    int counter = 0;
    while (fgets(line, 100, stream)) {
        char *subtoken, *tmp_line = line;
        for (int i = 0; i < n; ++i) {
            subtoken = strtok(tmp_line, ";");
            char *pEnd;
            tmp_line = strchr(line, ';');
            long res = strtol(subtoken, &pEnd, 10);
            if (input[counter] > INT_MIN || input[counter] < INT_MAX) {
                input[counter] = (int) res;
            } else {
                log_error("Input is not an integer --> Abort");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            log_trace("[MPI process %d] input[%d] = %d", rank, counter, input[counter]);
            counter++;
        }
    }

    fclose(stream);

//    if(MPI_File_close(&handle) != MPI_SUCCESS)
//    {
//        printf("[MPI process %d] Failure in closing the file.\n", rank);
//        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//    }

//    printf("[MPI process %d] File closed successfully.\n", rank);



    //check the size of col that each node gets
    log_trace("rank = %d, size = %d * m", rank, index_arr[rank]);

    // calculate how many cols each node gets
    index_calculation(index_arr, n, world_size);
//    display(rank, world_size, index_arr);

    //input matrix for each node:
    int rank_input[index_arr[rank]][m];

    //transposed input matrix for each node:
    int rank_input_t[m][index_arr[rank]];

    // compute the input matrix, and it's transpose,
    // which consists of the columns and all rows in that column:
    int rank_count = rank;
    for (int i = 0; i < rank; ++i) {
        if (i == 0) {
            rank_count = index_arr[i];
        } else {
            rank_count += index_arr[i];
        }
        log_trace("rank = %d, rank_count = %d", rank, rank_count);
    }
    for (int col_count = 0; col_count < index_arr[rank]; ++col_count) {
        for (int row_count = 0; row_count < m; ++row_count) {
            int tmp = input[(col_count + rank_count) + row_count * n];
            rank_input[col_count][row_count] = tmp;
            rank_input_t[row_count][col_count] = tmp;
            log_trace("rank-%d: rank_input[%d][%d] = %d", rank, col_count, row_count, rank_input[col_count][row_count]);
            log_trace("rank-%d: rank_input_t[%d][%d] = %d", rank, row_count, col_count,
                      rank_input_t[row_count][col_count]);
        }
    }

    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode 
    //  and store the result in rank_result
    int rank_result[m * m];
    // set all array entries to 0:
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            rank_result[i * m + j] = 0;
        }
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            for (int c = 0; c < index_arr[rank]; ++c) {
                log_trace("index_arr[%d] = %d", rank, index_arr[rank]);
                rank_result[row * m + col] = rank_input[c][row] * rank_input_t[col][c] + rank_result[row * m + col];
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * m + col]);
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * m + col]);
            }
        }
    }

    int counts[world_size];
    index_calculation(counts, m * m, world_size);
    //for (int i = 0; i < world_size; ++i) {
    //    counts[i] = index_arr[i];
    //}

    int reduction_result[counts[rank]];
    for (int i = 0; i < counts[rank]; ++i) {
        reduction_result[i] = 0;
    }

    //display(rank, (int) (sizeof(rank_result) / sizeof(rank_result[0])), rank_result);
    //display(rank, (int) (sizeof(reduction_result) / sizeof(reduction_result[0])), reduction_result);
    //MPI_Barrier(MPI_COMM_WORLD);

    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }

    //MPI_Barrier(MPI_COMM_WORLD);

    //display(rank, counts[rank], reduction_result);

    if (rank == 0) {
        log_debug("m = %d", m);
        int *buffer = (int *) calloc(m * m, sizeof(int));
        int displacements[world_size];
        displacements[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i-1] + counts[i-1];
        }
//        printf("counts:\n");
//        display(rank, world_size, counts);
//        printf("displacements:\n");
//        display(rank, world_size, displacements);

        MPI_Gatherv(reduction_result, counts[rank], MPI_INT, buffer, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        //Print the result:
        printf("Values gathered in the buffer on process %d:\n", rank);

        display(rank, m * m, buffer);
        free(buffer);
    } else {
        MPI_Gatherv(reduction_result, counts[rank], MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }


    // finish the program:
    MPI_Finalize();
    return EXIT_SUCCESS;
}

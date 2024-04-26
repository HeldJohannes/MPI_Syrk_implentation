#include "log.h"
#include "MPI_Syrk_implementation.h"
#include <cblas.h>
#include <float.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TRUE 1
#define FALSE 0


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
    log_set_level(LOG_TRACE);
    run_config *const config = malloc(sizeof(run_config));
    config->arg_0 = argv[0];

    // MPI initialization:
    int world_size;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int index_arr[world_size];


    // Parse the received ARGS to the run_config struct:
    if (argc != 6) {
        error_exit(rank, TRUE, argv[0], "To many or not enough input variables!");
    } else {
        parseInput(config, argc, argv, rank);
    }

    //input_matrix_in_array_form
    float input[config->m * config->n];
    read_input_file(rank, config, input);

    // calculate how many cols each node gets
    index_calculation(index_arr, config->n, world_size);

    //check the size of col that each node gets
    log_trace("rank = %d, size = %d * m", rank, index_arr[rank]);

    //input matrix for each node:
    float rank_input[index_arr[rank] * config->m];

    //transposed input matrix for each node:
    float rank_input_t[config->m * index_arr[rank]];

    // compute the input matrix, and it's transpose,
    // which consists of the columns and all rows in that column:
    computeInputAndTransposed(config, rank, index_arr, input, rank_input, rank_input_t);
    if (rank == 1) {
        printf("INPUT:\n");
        printf("index_arr[rank] = %d\n", index_arr[rank]);
        printf("config.m = %d\n", config->m);
        for (int i = 0; i < index_arr[rank] * config->m; ++i) {
            printf("rank_input[i = %d] = %f ", i, rank_input[i]);
            if (i % index_arr[rank] == 0) {
                printf("\n");
            }
        }
    }




    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode
    //  and store the result in rank_result
    float rank_result[config->m * config->m];
    syrk_1D(config, rank, index_arr, rank_input, rank_result);

    if (rank == 0) {
        printf("\nSYRK result:\n");
        for (int i = 0; i < config->m; ++i) {
            for (int j = 0; j < config->m; ++j) {
                printf("%f ", rank_result[i * config->m + j]);
            }
            printf("\n");
        }
    }


    int counts[world_size];
//    for (int i = 0; i < world_size; ++i) {
//        counts[i] = 450;
//    }
    index_calculation(counts, config->m * config->m, world_size);

    int reduction_result[counts[rank]];
    for (int i = 0; i < counts[rank]; ++i) {
        reduction_result[i] = 0;
    }

    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }

    if (rank == 0) {

        float *buffer = (float *) calloc(config->m * config->m, sizeof(float));
        int displacements[world_size];
        displacements[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i - 1] + counts[i];
        }
//        printf("counts:\n");
//        printResult(rank, world_size, counts);
//        printf("displacements:\n");
//        printResult(rank, world_size, displacements);

        MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, buffer, counts, displacements, MPI_FLOAT, 0,
                    MPI_COMM_WORLD);

        //Print the result:
        printf("Values gathered in the buffer on process %d:\n", rank);

        printResult(rank, config->m, config->m, buffer);
        printf("Print values on process %d:\n", rank);
        for (int i = 0; i < config->m; ++i) {
            for (int j = 0; j < config->m; ++j) {
                printf("%f ", buffer[i * config->m + j]);
            }
            printf("\n");
        }

        free(buffer);
        printf("Values print finished...");
    } else {
        MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }


    // finish the program:
    MPI_Finalize();
    free(config);
    return EXIT_SUCCESS;
}

void syrk_1D(run_config *s, const int rank, const int *index_arr, const float *rank_input, float *rank_result) {
    for (int i = 0; i < s->m * s->m; ++i) {
        rank_result[i] = 0;
    }
    cblas_ssyrk(
            CblasRowMajor,
            CblasUpper,
            CblasConjNoTrans,
            s->m,
            index_arr[rank],
            1.0f,
            rank_input,
            index_arr[rank],
            0.0f,
            rank_result,
            s->m);
}

void syrkIterative(run_config *s, int rank, const int *index_arr, const float rank_input[], const float rank_input_t[],
                   float rank_result[]) {
    // set all array entries to 0:
    log_debug("start syrkIterative:");
    for (int i = 0; i < s->m; ++i) {
        for (int j = 0; j < s->m; ++j) {
            rank_result[i * s->m + j] = 0;
        }
    }
    log_debug("after initial for loop");
    for (int row = 0; row < s->m; ++row) {
        //log_debug("outer for loop : row = %d; run_config.m = %d", row, s->m);
        for (int col = 0; col < s->m; ++col) {
            //log_debug("middle for loop : col = %d; run_config.n = %d", col, s->m);
            for (int c = 0; c < index_arr[rank]; ++c) {
                log_debug("c = %d", c);
                if (rank == 1)
                    log_debug("rank_result[%d] = %f * %f + %f;", row * s->m + col,
                              rank_input[c + row * index_arr[rank]],
                              rank_input_t[c * s->m + col], rank_result[row * s->m + col]);
                rank_result[row * s->m + col] =
                        rank_input[c + row * index_arr[rank]] * rank_input_t[c * s->m + col] +
                        rank_result[row * s->m + col];
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * s->m + col]);
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * s->m + col]);
            }
        }
    }
}


void computeInputAndTransposed(run_config *s, int rank,
                               const int *index_arr,
                               const float *input,
                               float *rank_input,
                               float *rank_input_t) {
    int rank_count = 0;
    for (int i = 0; i < rank; ++i) {
        rank_count += index_arr[i];
        log_info("rank = %d, rank_count = %d", rank, rank_count);
    }
    log_trace("index_arr[rank] = %d", index_arr[rank]);
    int i = 0;
    for (int row_count = 0; row_count < s->m; ++row_count) {
        for (int col_count = 0; col_count < index_arr[rank]; ++col_count) {
            float tmp = input[(col_count + rank_count) + row_count * s->n];
            log_trace("tmp = %f", tmp);
            log_trace("col_count = %d; row_count = %d", col_count, row_count);
            rank_input[i++] = tmp;
            log_debug("rank_input[col_count + row_count = %d] = %f", (col_count + rank_count) + row_count * s->n,
                      rank_input[(col_count + rank_count) + row_count * s->n]);
        }
    }
}

int read_input_file(const int rank, run_config *s, float *A) {
    FILE *stream = fopen(s->fileName, "r");

    if (stream == NULL) {
        error_exit(rank, FALSE, s->arg_0, "[MPI process %d] Can't open the file: %s", rank, s->fileName);
    }

    char *line = NULL;
    size_t len = 0;
    int counter = 0;
    while (getline(&line, &len, stream) != -1) {
        char *subtoken = strtok(line, ";");
        while (subtoken) {
            char *pEnd;
            float res = strtof(subtoken, &pEnd);
            if (res > FLT_MIN || res < FLT_MAX) {
                A[counter] = res;
            } else {
                log_error("Input is not an integer --> Abort");
                log_error("%f", res);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            log_trace("[MPI process %d] input[%d] = %f", rank, counter, A[counter]);
            counter++;
            subtoken = strtok(NULL, ";");
        }
    }

    fclose(stream);
    return EXIT_SUCCESS;
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

void printResult(int rank, int len, int cols, float array[]) {
    printf("Print values on process %d:\n", rank);
    FILE *file = fopen("result.csv", "w");
    //char *string = "[MPI process %d] ";
    //fprintf(file ,string , rank);
    for (int j = 0; j < len; ++j) {
        for (int i = j * cols; i < (j + 1) * cols; ++i) {
            if (i == j * cols + cols - 1) {
                //fprintf(file, "%d", array[i]);
                fprintf(file, "%f ", array[i]);
            } else {
                //fprintf(file, "%d; ", array[i]);
                fprintf(file, "%f; ", array[i]);
            }
        }
        //fprintf(file, "\n");
        fprintf(file, "\n");
    }
    printf("\n");
}

void parseInput(run_config *s, int argc, char **argv, int rank) {
    log_trace("parseInput(run_config *s, int argc (= %d), char **argv, int rank (= %d))", argc, rank);

    //total_row_number
    s->m = -1;
    //total_col_number
    s->n = -1;

    int opt;
    char *end;
    while ((opt = getopt(argc, argv, "m:n:")) != -1) {
        switch (opt) {
            case 'm':
                s->m = (int) strtol(optarg, &end, 10);
                break;
            case 'n':
                s->n = (int) strtol(optarg, &end, 10);
                break;
            default:
            case '?':
                error_exit(rank, TRUE, argv[0], "wrong usage: option %c doesn't exist", opt);
        }
    }

    log_trace("m = %d; n = %d", s->m, s->n);
    if (s->m == -1 || s->n == -1) {
        error_exit(rank, TRUE, argv[0], "parameters m (= %d) and n (= %d) have to be bigger than 0", s->m,
                   s->n);
    }

    log_trace("optind = %d, argc = %d", optind, argc);
    s->fileName = argv[optind];
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

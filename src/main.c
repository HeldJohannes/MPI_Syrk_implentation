#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include "log.h"
#include "MPI_Syrk_implementation.h"

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
    log_set_level(LOG_FATAL);
    static run_config config;
    int world_size, rank;

    MPI_Init(&argc, &argv);

//    int debug = 0;
//    while (!debug)
//        sleep(5);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int index_arr[world_size];
    if (argc <= 5) {
        error_exit(rank, argv[0], "To many or not enough input variables!");
    }

    parseInput(&config, argc, argv, rank);

    log_info("Size = %d (SIZE_MAX = %zu) => %zu", config.m * config.n, SIZE_MAX, config.m * config.n * sizeof(float));
    //input_matrix_in_array_form
    float *input = (float *) calloc(config.m * config.n, sizeof(float));


    //read the input matrix A:
    read_input_file(rank, &config, input);

    //check the size of col that each node gets
    log_trace("rank = %d, size = %d * m", rank, index_arr[rank]);

    // calculate how many cols each node gets
    index_calculation(index_arr, config.n, world_size);

    //input matrix for each node:
    log_info("index_arr[rank] => index_arr[%d] = %d", rank, index_arr[rank]);
    log_info("size = %d", config.m);
    float **rank_input = (float **) calloc(config.m, sizeof(float *));

    //transposed input matrix for each node:
    float *rank_input_t = (float *) calloc(config.m * index_arr[rank], sizeof(float));

    // compute the input matrix, and it's transpose,
    // which consists of the columns and all rows in that column:
    computeInputAndTransposed(&config, rank, index_arr, input, rank_input, rank_input_t);
    log_info("computeInputAndTransposed Successful");

    double start = MPI_Wtime();

    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode 
    //  and store the result in rank_result
    float *rank_result = (float *) calloc(config.m * config.m, sizeof(float));
    syrkIterative(&config, rank, index_arr, rank_input, rank_input_t, rank_result);
    log_info("syrk Successful");

    int counts[world_size];
    index_calculation(counts, config.m * config.m, world_size);

    int reduction_result[counts[rank]];
    for (int n = 0; n < counts[rank]; ++n) {
        reduction_result[n] = 0;
    }

    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }

    if (rank == 0) {
        log_debug("m = %d", config.m);
        float *buffer = (float *) calloc(config.m * config.m, sizeof(float));
        int *displacements = (int *) calloc(world_size, sizeof(int));
        displacements[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displacements[i] = displacements[i - 1] + counts[i - 1];
        }
//        printf("counts:\n");
//        printResult(rank, world_size, counts);
//        printf("displacements:\n");
//        printResult(rank, world_size, displacements);

        int status = MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, buffer, counts, displacements, MPI_FLOAT, 0,
                    MPI_COMM_WORLD);

        if (status != MPI_SUCCESS) {
            log_error("MPI_Gatherv returned %d", status);
        }


        //double runtime = MPI_Wtime();

        log_info("before printf");
        //Print the result:
        printf("Values gathered in the buffer on process %d:", rank);
        printf("The process took %f seconds to run.", (MPI_Wtime() - start));

        //printResult(&config, config.m, buffer);


        free(buffer);
        log_info("[if rank == 0]: Successfully freed -> buffer...");
        free(displacements);
        log_info("[if rank == 0]: Successfully freed -> displacements...");
    } else {
        int status;
        if ((status = MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD)) != MPI_SUCCESS) {
            log_error("MPI_Gatherv returned %d", status);
        }
    }

    free(input);
    log_info("Successfully freed the buffer -> input");
    free(rank_input);
    log_info("Successfully freed the buffer -> rank_input");
    free(rank_input_t);
    log_info("Successfully freed the buffer -> rank_input_t");
    free(rank_result);
    log_info("Successfully freed the buffer -> rank_result");

    log_info("finish the program:");
    // finish the program:
    int status;
    if ((status = MPI_Finalize()) != MPI_SUCCESS) {
        log_error("MPI Failed with %d", status);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void syrkIterative(run_config *s, int rank, int *index_arr, float **rank_input, float rank_input_t[],
                   float *rank_result) {
    // set all array entries to 0:
    log_debug("start syrkIterative:");
//    for (int i = 0; i < s->m; ++i) {
//        for (int j = 0; j < s->m; ++j) {
//            rank_result[i * s->m + j] = 0;
//        }
//    }
    log_debug("after initial for loop");
    // for each result row:
    for (int row = 0; row < s->m; ++row) {
        //log_debug("outer for loop : row = %d; run_config.m = %d", row, s->m);
        // for each result column
        for (int col = 0; col < s->m; ++col) {
            //log_debug("middle for loop : col = %d; run_config.n = %d", col, s->m);
            // run for slice of the input:
            for (int c = 0; c < index_arr[rank]; ++c) {
                log_debug("c = %d", c);
                if (rank == 1)
                    log_debug("rank_result[%d] = %f * %f + %f;", row * s->m + col,
                              rank_input[c + row * index_arr[rank]],
                              rank_input_t[c * s->m + col], rank_result[row * s->m + col]);

                rank_result[row * s->m + col] = *(rank_input[row] + c) * rank_input_t[c * s->m + col] + rank_result[row * s->m + col];

                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * s->m + col]);
                log_trace("rank = %d; i = %d j = %d; result = %d", rank, row, col, rank_result[row * s->m + col]);
            }
        }
    }
}

void computeInputAndTransposed(run_config *s, int rank, int *index_arr, float *input, float **rank_input,
                               float *rank_input_t) {
    int rank_count = 0;
    for (int i = 0; i < rank; ++i) {
        rank_count += index_arr[i];
        log_info("rank = %d, rank_count = %d", rank, rank_count);
    }
    log_trace("index_arr[rank] = %d", index_arr[rank]);

    for (int row_count = 0; row_count < s->m; row_count++) {
        //for (int col_count = 0; col_count < index_arr[rank]; ++col_count) {
        //float tmp = input[(col_count + rank_count) + row_count * s->n];
        log_info("read index = %d", (rank_count) + row_count * s->n);
        //log_trace("tmp = %f", tmp);
        //log_trace("col_count = %d; row_count = %d", col_count, row_count);
        float *tmp = &input[(rank_count) + row_count * s->n];
        log_debug("&input[(rank_count) + row_count * s->n] = %p => %f", tmp, *tmp);
        rank_input[row_count] = tmp;
        log_debug("rank_input[row_count] = %p => %f", rank_input[row_count], *rank_input[row_count]);
        //log_trace("rank_input[col_count + row_count = %d + %d (%d)] = %f", col_count, row_count, (col_count + rank_count) + row_count * s->n, rank_input[i]);
        //}
    }
    log_info("for loop success");

    transposeMatrix(s->m, index_arr[rank], rank_input, rank_input_t);
}

int read_input_file(const int rank, run_config *s, float *A) {
    FILE *stream = fopen(s->fileName, "r");

    if (stream == NULL) {
        printf("[MPI process %d] Failure in opening the file.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
                log_error("Input is not an float --> Abort");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            log_trace("[MPI process %d] input[%d] = %d", rank, counter, A[counter]);
            counter++;
            subtoken = strtok(NULL, ";");
        }
    }

    fclose(stream);
    return EXIT_SUCCESS;
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

void printResult(run_config *s, int cols, float *array) {
    FILE *file;
    if (s->result_File != NULL) {
        log_info("Printing result to set file = %s", s->result_File);
        file = fopen(s->result_File, "w");
    } else {
        log_info("Printing result to default file = result.csv");
        file = fopen("result.csv", "w");
    }


    //char *string = "[MPI process %d] ";
    //fprintf(file ,string , rank);
    for (int j = 0; j < cols; ++j) {
        for (int i = j * cols; i < j * cols + cols; ++i) {
            log_debug("%f", array[i]);
            if (i == j * cols + cols - 1) {
                fprintf(file, "%f", array[i]);
            } else {
                fprintf(file, "%f; ", array[i]);
            }

        }
        fprintf(file, "\n");
    }

    log_info("Finished printResults()");
}

/**
 * Prints usage information for the program.
 *
 * @param prog_name Name of the program
 */
void print_usage(char *prog_name) {
    fprintf(stderr, "Usage: \"mpiexec -np <CORES> %s -o <output_file> -m <ROWS> -n <COLS> <input_file> \"", prog_name);
}

void parseInput(run_config *s, int argc, char **argv, int rank) {

    log_trace("Enter parseInput");
    //total_row_number
    s->m = -1;
    //total_col_number
    s->n = -1;

    int opt;
    char *end;
    while ((opt = getopt(argc, argv, "m:n:o:")) != -1) {
        switch (opt) {
            case 'm':
                s->m = (int) strtol(optarg, &end, 10);
                break;
            case 'n':
                s->n = (int) strtol(optarg, &end, 10);
                break;
            case 'o':
                s->result_File = optarg;
                break;
            default:
            case '?':
                print_usage(argv[0]);
                if (optopt == '?') {
                    exit(0);
                }
                error_exit(rank, argv[0], "wrong usage: option %c doesn't exist", opt);
        }
    }

    log_trace("m = %d; n = %d", s->m, s->n);
    if (s->m == -1 || s->n == -1) {
        error_exit(rank, argv[0], "parameters m (= %d) and n (= %d) have to be bigger than 0", s->m, s->n);
    }

    log_trace("optind = %d, argc = %d", optind, argc);
    s->fileName = argv[optind];
    log_trace("Exit parseInput");
}

void error_exit(int rank, char *name, const char *msg, ...) {
    if (rank == 0) {

        char buf[16];
        time_t t = time(NULL);
        buf[strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&t))] = '\0';
        fprintf(stderr, "[%s] %s %-5s : ", name, buf, "ERROR");

        va_list ap;
        va_start(ap, msg);
        vfprintf(stderr, msg, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
}

void transposeMatrix(int m, int n, float **matrix, float *result) {

    for (int i = 0; i < m; i++) {

        //log_info("tmp[i] (%d)", i);
        //log_info("tmp[i] = %p", tmp[i]);
        //log_info("tmp[i] => %f", *(tmp[i] + 0));

        for (int j = 0; j < n; j++) {

            //if(i >= 2) log_info("Write Result to %p / from %p", &result[j * m + i], (matrix[i] + j));

            float value = *(matrix[i] + j);
            result[j * m + i] = value;

            //log_info("rank_input_t[%d] = %f", j * m + i, result[j * m + i]);
        }

    }
}

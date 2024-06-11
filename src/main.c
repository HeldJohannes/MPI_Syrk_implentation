#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include "log.h"
#include <openblas/cblas.h>
#include "MPI_Syrk_implementation.h"



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

    int *index_arr = (int *) calloc(world_size, sizeof(int));
    if (argc <= 5) {
        error_exit(rank, argv[0], "To many or not enough input variables!");
    }

    parseInput(&config, argc, argv, rank);


    log_info("Size = %d (SIZE_MAX = %zu) => %zu", config.m * config.n, SIZE_MAX, config.m * config.n * sizeof(float));
    //input_matrix_in_array_form
    if (rank == 0) {
        log_debug("[int] config.m * config.n = %d", config.m * config.n);
        log_debug("[long] config.m * config.n = %ld", config.m * config.n);
    }
    float *input = (float *) calloc(config.m * config.n, sizeof(float));
    if (!input) {
        log_debug("Memory allocation failed for input");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    //read the input matrix A:
    if (rank == 0) {
        read_input_file(rank, &config, input);
    }
    MPI_Bcast(input, config.m * config.n, MPI_FLOAT, 0, MPI_COMM_WORLD);


    //check the size of col that each node gets
    log_trace("rank = %d, size = %d * m", rank, index_arr[rank]);

    // calculate how many cols each node gets
    index_calculation(index_arr, config.n, world_size);

    //input matrix for each node:
    log_info("index_arr[rank] => index_arr[%d] = %d", rank, index_arr[rank]);
    log_info("size = %d", config.m);
    float **rank_input = (float **) calloc(config.m, sizeof(float *));
    for (int i = 0; i < config.m; ++i) {
        rank_input[i] = (float *) calloc(index_arr[rank], sizeof(float));
        if (!rank_input[i]) {
            log_fatal("Memory allocation failed for rank_input[%d]", i);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    //transposed input matrix for each node:
    if (rank == 0) {
        log_debug("[int] config.m * index_arr[rank] = %d", config.m * index_arr[rank]);
        log_debug("[long] config.m * index_arr[rank] = %ld", config.m * index_arr[rank]);
    }
    float *rank_input_t = (float *) calloc(config.m * index_arr[rank], sizeof(float));
    if (!rank_input_t) {
        log_fatal("Memory allocation failed for input");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // compute the input matrix, and it's transpose,
    // which consists of the columns and all rows in that column:
    computeInputAndTransposed(&config, rank, index_arr, input, rank_input, rank_input_t);



    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode 
    //  and store the result in rank_result
    log_trace(" config.m * config.m < INT_MAX == %d", config.m * config.m < INT_MAX);
    long test = config.m * config.m;
    log_trace("Test config.m = %d; config.m * config.m = %lld", config.m, test);
    float *rank_result = (float *) calloc(test, sizeof(float));
    if (!rank_result) {
        log_fatal("[processor %d] Memory allocation failed for input with errno", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double start = MPI_Wtime();
    switch (ALGO) {
        case 0:
            syrkIterative(&config, rank, index_arr, rank_input, rank_input_t, rank_result);
            break;
        case 1:
            improved_syrkIterative(&config, rank, index_arr, rank_input, rank_input_t, rank_result);
            break;
        case 2:
            syrk_withOpenBLAS(&config, rank, index_arr, rank_input, rank_result);
            break;
        default:
            log_fatal("no SYRK operator selected --> error ALOG %d not in [0..2]", ALGO);
            error_exit(rank, argv[0], "no SYRK operator selected");
    }
    log_info("syrk Successful");
    log_fatal("Syrk algo(%d) took %f sec", ALGO, MPI_Wtime() - start);


    int *counts = (int *) calloc(world_size, sizeof(int));
    index_calculation(counts, config.m * config.m, world_size);
    log_info("index_calculation Successful");

    int *reduction_result = (int *) calloc(counts[rank], sizeof(int));

    double start_mpi_reduce_scatter = MPI_Wtime();

    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }
    log_info("MPI_Reduce_scatter Successful");
    double runtime_mpi_reduce_scatter = MPI_Wtime() - start_mpi_reduce_scatter;
    log_debug("[rank %d]: MPI_Reduce_scatter took %f sec",rank, runtime_mpi_reduce_scatter);

    if (rank == 0) {
        log_debug("m = %d", config.m);
        log_debug("[int] config.m * config.m = %d", config.m * config.m);
        log_debug("[long] config.m * config.m = %ld", config.m * config.m);
        float *buffer = (float *) calloc(config.m * config.m, sizeof(float));
        if (!buffer) {
            log_error("Memory allocation failed for input with errno");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int *displacements = (int *) calloc(world_size, sizeof(int));
        if (!displacements) {
            log_error("Memory allocation failed for input with errno");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
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


        double runtime = MPI_Wtime() - start;

        //Print the result:
        printf("Values gathered in the buffer on process %d:\n", rank);
        printf("The process took %f seconds to run.\n", runtime);

        printResult(&config, config.m, buffer);


        free(buffer);
        log_info("[if rank == 0]: Successfully freed -> buffer...");
        free(displacements);
        log_info("[if rank == 0]: Successfully freed -> displacements...");
    } else {
        int status;
        status = MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) {
            log_error("MPI_Gatherv returned %d", status);
        }
    }

    free(index_arr);
    log_info("Successfully freed the buffer -> index_arr");
    free(reduction_result);
    log_info("Successfully freed the buffer -> reduction_result");
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
    log_debug("start syrkIterative:");
    log_info("index_arr[rank = %d] == %d", rank, index_arr[rank]);
    // for each result row:
    for (long row = 0; row < s->m; ++row) {
        // for each result column
        for (long col = 0; col < s->m; ++col) {
            // run for slice of the input:
            for (long c = 0; c < index_arr[rank]; ++c) {
                rank_result[row * s->m + col] += rank_input[row][c] * rank_input_t[c * s->m + col];
            }
        }
    }
}

void improved_syrkIterative(run_config *s, int rank, const int *index_arr, float **rank_input, const float rank_input_t[],
                       float rank_result[]) {

    for (int row = 0; row < s->m; ++row) {
        //log_debug("outer for loop : row = %d; run_config.m = %d", row, s->m);
        for (int col = row; col < s->m; ++col) {
            //log_debug("middle for loop : col = %d; run_config.n = %d", col, s->m);
            for (int c = 0; c < index_arr[rank]; ++c) {

                rank_result[row * s->m + col] += rank_input[row][c] * rank_input_t[c * s->m + col];
            }
        }
    }
}

void syrk_withOpenBLAS(run_config *config, int rank, int *index_arr, float **rank_input, float *rank_result) {
    cblas_ssyrk(
            CblasRowMajor,
            CblasUpper,
            CblasConjNoTrans,
            config->m,
            index_arr[rank],
            1.0f,
            *rank_input,
            index_arr[rank],
            0.0f,
            rank_result,
            config->m);
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
        //log_info("read index = %d", (rank_count) + row_count * s->n);
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
//    MPI_File mpiFile;
//    if (MPI_File_open(MPI_COMM_WORLD, s->fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpiFile)) {
//        printf("[MPI process %d] Failure in opening the file.\n", rank);
//        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//    }
//
//    MPI_Offset filesize;
//    MPI_File_get_size(mpiFile, &filesize);
//    printf("[MPI process %d] File size == %lld\n", rank, filesize);
//
//    MPI_File_read(mpiFile, )
//
//
//    MPI_File_close(&mpiFile);

    FILE *stream = fopen(s->fileName, "r");

    if (stream == NULL) {
        printf("[MPI process %d] Failure in opening the file.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char *line = NULL;
    size_t len = 0;
    int counter = 0;
    while (1) {
        ssize_t n = getline(&line, &len, stream);
        log_error("ssize_t n = %d", n);
        if (n == -1) break;
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
            //log_trace("[MPI process %d] input[%d] = %d", rank, counter, A[counter]);
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
        file = fopen("syrk_result.csv", "w");
    }


    //char *string = "[MPI process %d] ";
    //fprintf(file ,string , rank);
    for (int j = 0; j < cols; ++j) {
        for (int i = j * cols; i < j * cols + cols; ++i) {
            log_debug("%f", array[i]);
            if (i == j * cols + cols - 1) {
                fprintf(file, "%0.0f", array[i]);
            } else {
                fprintf(file, "%0.0f; ", array[i]);
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
    while ((opt = getopt(argc, argv, "a:m:n:o:")) != -1) {
        switch (opt) {
            case 'a':
                ALGO = (int) strtol(optarg, &end, 10);
                break;
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

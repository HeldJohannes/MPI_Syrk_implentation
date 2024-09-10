#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <float.h>
#include "MPI_Syrk_implementation.h"
#include "two_d_syrk.h"

_Bool PRINT_RESULT = false;
int ALGO = 0;

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
    static run_config config;
    int world_size, rank;

    MPI_Init(&argc, &argv);
//    int debug = 0;
//    while (!debug)
//        sleep(5);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    config.world_size = world_size;

    if (argc <= 5) {
        error_exit(rank, argv[0], "To many or not enough input variables!");
    }

    parseInput(&config, argc, argv, rank);


    log_debug("Size = %d (SIZE_MAX = %zu) => %zu", config.m * config.n, SIZE_MAX, config.m * config.n * sizeof(float));
    //input_matrix_in_array_form
    float *input_array = (float *) calloc(config.m * config.n, sizeof(float ));
    if (input_array == NULL) {
        log_fatal("Memory allocation failed for input");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    float **input = (float **) calloc(config.m, sizeof(float *));
    for (int i = 0; i < config.m; ++i) {
        input[i] = &(input_array[config.n * i]);
    }


    int index_arr_rank;
    //int cumulate_index_arr_rank;
    int *index_arr = (int *) calloc(world_size, sizeof(int));
    if (!index_arr) {
        log_fatal("Memory allocation failed for cumulate_index_arr");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int *cumulate_index_arr = (int *) calloc(world_size, sizeof(int));
    if (!cumulate_index_arr) {
        log_fatal("Memory allocation failed for cumulate_index_arr");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // read the input matrix A
    // and compute the index_array
    if (rank == 0) {
        read_input_file(rank, &config, input);
        // calculate how many cols each node gets
        index_calculation(index_arr, config.n, world_size);

        int rank_count = 0;
        for (int i = 0; i < world_size; ++i) {
            assert(cumulate_index_arr + i != NULL);
            cumulate_index_arr[i] = rank_count;
            assert(index_arr + i != NULL);
            rank_count += index_arr[i];
        }
    }

    // send index_arr[rank] to all processors and work with index_arr_rank
    MPI_Scatter(index_arr, 1, MPI_INT, &index_arr_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // send cumulate_index_arr[rank] to all processors and work with cumulate_index_arr_rank
    //MPI_Scatter(cumulate_index_arr, 1, MPI_INT, &cumulate_index_arr_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //input matrix for each node:
    float **rank_input;
    if (ALGO != 3) {
        rank_input = (float **) calloc(config.m, sizeof(float *));
        for (int i = 0; i < config.m; ++i) {
            rank_input[i] = (float *) calloc(index_arr_rank, sizeof(float));
            if (!rank_input[i]) {
                log_fatal("Memory allocation failed for rank_input[%d]", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        for (int i = 0; i < config.m; ++i) {
            // for each row split and distribute across all processors
            MPI_Scatterv(input[i], index_arr, cumulate_index_arr, MPI_FLOAT, rank_input[i], index_arr_rank, MPI_FLOAT,
                         0,
                         MPI_COMM_WORLD);
        }
    } else {

        //todo find a better solution:
        MPI_Bcast(input_array, config.m * config.n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        /*
         * Step 1:
         *
         * Create new Communicators Based on Q_i to take advantage of MPI_SCATTER to split A_i as A_i^(k).
         * note: this step might not be necessary but I couldn't find a better current solution
         */
        MPI_Group main_group;
        MPI_Comm_group(MPI_COMM_WORLD, &main_group);
        // create the mpi - groups:
        MPI_Group *pMpiGroups = (MPI_Group *) malloc(config.c * config.c * sizeof (MPI_Group ));
        if (!pMpiGroups) {
            log_fatal("Memory allocation failed for pMpiGroups", 0);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // create mpi Communicators
        MPI_Comm *pMpiCommunicators = (MPI_Comm *) malloc(config.c * config.c * sizeof (MPI_Comm ));
        if (!pMpiCommunicators) {
            log_fatal("Memory allocation failed for pMpiCommunicators", 0);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int *Q_i = (int *) calloc(config.c + 1, sizeof(int));
        for (int i = 0; i < config.c * config.c; ++i) {
            calculate_Q_i(Q_i, i, config.c);
            int ret = MPI_Group_incl(main_group, config.c + 1, Q_i,&pMpiGroups[i]);
            if (ret != MPI_SUCCESS) {
                log_fatal("MPI_Group_incl failed for group %d", i);
                MPI_Abort(MPI_COMM_WORLD, ret);
            }
            assert(pMpiGroups[i] != NULL);
            int err = MPI_Comm_create(MPI_COMM_WORLD, pMpiGroups[i], &pMpiCommunicators[i]);
            if (err != MPI_SUCCESS) {
                log_error("MPI_Comm_create_group error");
            }
            assert(pMpiCommunicators[i] != NULL);
        }


        /*
         * Step 2:
         *
         * Spilt the input:
         */

        // Allocate the arrays:
        // each node gets c blocks of A_i^(k)
        int row_block_height = config.m / (config.c * config.c);
        int row_block_length = (config.n / (config.c + 1));
        //log_info("[rank = %d] row_block_length = %d", rank, row_block_length);
        rank_input = (float **) calloc(config.c * row_block_height, sizeof (float *));
        //int row_block_size = row_block_height * row_block_length;
        for (int i = 0; i < config.c * row_block_height; ++i) {
            rank_input[i] = (float *) calloc(row_block_length, sizeof (float ));
            if (!rank_input[i]) {
                log_fatal("Memory allocation failed for rank_input[%d]", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        int *counts = (int *) calloc(config.c * config.c, sizeof(int ));

        //config.world_size = config.c +1;


        // Distribute the rows with MPI_Scatter:
        for (int i = 0; i < config.c * config.c; ++i) {
            calculate_Q_i(Q_i, i, config.c);
            for (int j = 0; j < row_block_height; ++j) {
                //int test_world_size = world_size;
                assert(pMpiCommunicators[i] != NULL);
                //MPI_Comm_size(pMpiCommunicators[i], &test_world_size);
                //log_info("[rank = %d] world size = %d (row_block_height == %d)", rank, test_world_size, row_block_height);
                if (pMpiCommunicators[i] != MPI_COMM_NULL) {
                    MPI_Scatter(input[i * row_block_height + j], row_block_length, MPI_FLOAT, rank_input[((counts[rank]) * row_block_height) + j],
                                row_block_length, MPI_FLOAT, Q_i[0], pMpiCommunicators[i]);
                }
            }
            for (int j = 0; j < config.c +1; ++j) {
                assert(Q_i[j] < config.world_size);
                counts[Q_i[j]] += 1;
            }
        }

        free(counts);
        free(Q_i);
        //print the rank_input:

    }

    // free index_arr and cumulate_index_arr because they are no longer needed
    free(index_arr);
    free(cumulate_index_arr);

    // input no longer needed
    free(input_array);
    log_debug("Successfully freed the buffer -> input");


    //transposed input matrix for each node:
    float **rank_input_t = (float **) calloc(index_arr_rank, sizeof(float *));
    if (ALGO != 3) {
        for (int i = 0; i < index_arr_rank; ++i) {
            rank_input_t[i] = (float *) calloc(config.m, sizeof(float));
            if (!rank_input_t[i]) {
                log_fatal("Memory allocation failed for input");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }


        // compute the input matrix, and it's transpose,
        // which consists of the columns and all rows in that column:
        //computeInputAndTransposed(&config, rank, index_arr_rank, cumulate_index_arr_rank, input, rank_input, rank_input_t);

        transposeMatrix(config.m, index_arr_rank, rank_input, rank_input_t);
    }

    // SYRK:
    // Compute the result matrix for each node which gets
    //  summed up by the MPI_Reduce_scatter() methode 
    //  and store the result in rank_syrk_result
    float *rank_syrk_result = (float *) calloc((long) config.m * config.m, sizeof(float));
    if (!rank_syrk_result) {
        log_fatal("[processor %d] Memory allocation failed for input with errno", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    // Synchronize before starting time
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    switch (ALGO) {
        case 0:
            syrkIterative(&config, rank, index_arr_rank, rank_input, rank_input_t, rank_syrk_result);
            break;
        case 1:
            improved_syrkIterative(&config, rank, index_arr_rank, rank_input, rank_input_t, rank_syrk_result);
            break;
        case 2:
            syrk_withOpenBLAS(&config, rank, index_arr_rank, rank_input, rank_syrk_result);
            break;
        case 3:
            two_d_syrk(&config, rank, rank_syrk_result, rank_input);
            break;
        default:
            log_fatal("no SYRK operator selected --> error ALOG %d not in [0..2]", ALGO);
            error_exit(rank, argv[0], "no SYRK operator selected");
    }
    // Synchronize again before obtaining the time
    //MPI_Barrier(MPI_COMM_WORLD);
    //log_info("Syrk algo(%d) took %f sec", ALGO, MPI_Wtime() - start);
    log_debug("Successfully freed the buffer -> cumulate_index_arr");
    free(rank_input);
    log_debug("Successfully freed the buffer -> rank_input");
    free(rank_input_t);
    log_debug("Successfully freed the buffer -> rank_input_t");


    int *counts = (int *) calloc(world_size, sizeof(int));
    index_calculation(counts, (long) config.m * config.m, world_size);

    float *reduction_result = (float *) calloc(counts[rank], sizeof(float));

    double start_mpi_reduce_scatter = MPI_Wtime();

//    float *buffer = NULL;
//    if (rank == 0) {
//        buffer = (float *) calloc(config.m * config.m, sizeof(float));
//    }
    //int result = MPI_Reduce(rank_syrk_result, buffer, config.m * config.m, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    int result = MPI_Reduce_scatter(rank_syrk_result, reduction_result, counts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        log_error("MPI_Reduce_scatter returned exit coed %d", result);
    }

    double runtime_mpi_reduce_scatter = MPI_Wtime() - start_mpi_reduce_scatter;
    //log_debug("[rank %d]: MPI_Reduce_scatter took %f sec", rank, runtime_mpi_reduce_scatter);
    log_debug("[rank %d]: MPI_Reduce took %f sec", rank, runtime_mpi_reduce_scatter);

    if (rank == 0) {
        double runtime = MPI_Wtime() - start;
        printf("The process took %f seconds to run.\n", runtime);

        log_debug("m = %d", config.m);
        log_debug("[int] config.m * config.m = %d", config.m * config.m);
        log_debug("[long] config.m * config.m = %ld", config.m * config.m);
        float *buffer = (float *) calloc(config.m * config.m, sizeof(float));

        if (!buffer) {
            log_fatal("Memory allocation failed for input with errno");
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
        int status = MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, buffer, counts, displacements, MPI_FLOAT,
                                 0,
                                 MPI_COMM_WORLD);

        if (status != MPI_SUCCESS) {
            log_error("MPI_Gatherv returned %d", status);
        }




        //Print the result:
        log_info("Values gathered in the buffer on process %d\n", rank);

        if (PRINT_RESULT) {
            // No synchronization needed because only processor 0 operates here
            double start_print_results = MPI_Wtime();

            printResult(&config, config.m, buffer);

            double runtime_print_results = MPI_Wtime() - start_print_results;
            log_info("runtime_print_results = %f", runtime_print_results);
        }

        free(buffer);
        log_debug("[if rank == 0]: Successfully freed -> buffer...");
        free(displacements);
        log_debug("[if rank == 0]: Successfully freed -> displacements...");
    } else {
        int status;
        status = MPI_Gatherv(reduction_result, counts[rank], MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) {
            log_error("MPI_Gatherv returned %d", status);
        }
    }

    free(reduction_result);
    log_debug("Successfully freed the buffer -> reduction_result");

    free(rank_syrk_result);
    log_debug("Successfully freed the buffer -> rank_syrk_result");

    log_info("[rank %d] finished the program --> MPI_Finalize()", rank);
    // finish the program:
    int status;
    if ((status = MPI_Finalize()) != MPI_SUCCESS) {
        log_error("MPI Failed with %d", status);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}



void computeInputAndTransposed(run_config *s, int rank, int index_arr_rank, int cum_index_arr_rank, float **input,
                               float **rank_input,
                               float **rank_input_t) {
    log_trace("[rank %d] computeInputAndTransposed()", rank);
    for (int row = 0; row < s->m; row++) {
        for (int col = 0; col < index_arr_rank; ++col) {
            float tmp = input[row][cum_index_arr_rank + col];
            rank_input[row][col] = tmp;
            log_debug("&input[(rank_count) + row_count * s->n] = %p => %f", &tmp, tmp);
            log_debug("rank_input[row_count] = %p => %f", rank_input[row], *rank_input[row]);
        }
    }
    log_debug("for loop success");

    transposeMatrix(s->m, index_arr_rank, rank_input, rank_input_t);
}

int read_input_file(const int rank, run_config *s, float **A) {
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
        log_fatal("[MPI process %d] Failure in opening the file.", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char *line = NULL;
    size_t len = 0;
    int row = 0;
    int col = 0;
    while (1) {
        ssize_t n = getline(&line, &len, stream);
        log_debug("ssize_t n = %d", n);
        if (n == -1) break;
        char *subtoken = strtok(line, ";");
        while (subtoken) {
            char *pEnd;
            float res = strtof(subtoken, &pEnd);
            if (res > FLT_MIN || res < FLT_MAX) {
                A[row][col] = res;
            } else {
                log_error("Input is not an float --> Abort");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            //log_trace("[MPI process %d] input[%d] = %d", rank, row, A[row]);
            col++;
            subtoken = strtok(NULL, ";");
        }
        col = 0;
        row++;
    }

    fclose(stream);
    return EXIT_SUCCESS;
}

void index_calculation(int *arr, long n, int p) {
    long input_size = n / p;

    long rest = n % p;
    log_trace("rest = %d", rest);

    for (int i = 0; i < p; ++i) {
        assert(arr + i != NULL);
        arr[i] = (int) input_size;
        if (i < rest) {
            arr[i] += 1;
        }
    }
}

void printArray(int row, int cols, const float *array, FILE *file) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < cols; ++j) {
            log_debug("array[%d][%d] = %f", i, j, array[i * cols + j]);
            if (j == cols - 1) {
                fprintf(file, "%0.0f", array[i * cols + j]);
            } else {
                fprintf(file, "%0.0f; ", array[i * cols + j]);
            }

        }
        fprintf(file, "\n");
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

    printArray(cols, cols, array, file);

    log_debug("Finished printResults()");
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
    while ((opt = getopt(argc, argv, "a:m:n:o:c:")) != -1) {
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
            case 'c':
                s->c = (int) strtol(optarg, &end, 10);
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

void transposeMatrix(long m, long n, float **matrix, float **result) {
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            assert(matrix[i] + j != NULL);
            assert(result[j] + i != NULL);
            result[j][i] = matrix[i][j];
        }
    }
}

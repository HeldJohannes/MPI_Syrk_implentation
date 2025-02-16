#include "MPI_Syrk_implementation.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <float.h>
#include "log.h"

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
                s->algo = (int) strtol(optarg, &end, 10);
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
                fprintf(stderr, "wrong usage: option %c doesn't exist", optopt);
                //print_usage(argv[0]);
                exit(EXIT_FAILURE); 
        }
    }

    log_trace("m = %d; n = %d", s->m, s->n);

    if (s->m == -1)
    {
        fprintf(stderr, "missing parameter m\n");
        exit(EXIT_FAILURE);
    }
    if (s->n == -1)
    {
        fprintf(stderr, "missing parameter n\n");
        exit(EXIT_FAILURE);
    }
    if (s->m <= -1 || s->n <= -1) {
        fprintf(stderr, "parameters m (= %d) and n (= %d) have to be bigger than 0\n", s->m, s->n);
        exit(EXIT_FAILURE);
    }

    log_trace("optind = %d, argc = %d", optind, argc);
    if (optind < argc) {
        s->fileName = argv[optind];
    } else {
        fprintf(stderr, "missing input file name\n Generate random input\n");
    }
    log_trace("Exit parseInput");
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

void transposeMatrix(long m, long n, float **matrix, float **result) {
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            assert(matrix[i] + j != NULL);
            assert(result[j] + i != NULL);
            result[j][i] = matrix[i][j];
        }
    }
}

void generate_input(run_config *s, float **A) {
    static int seeded = 0;
    if (!seeded) {
        srandom((unsigned int) time(NULL));
        seeded = 1;
    }

    for (int i = 0; i < s->m; i++) {
        for (int j = 0; j < s->n; j++) {
            A[i][j] = ((float) random() / RAND_MAX) * 10.0;
        }
    }
}

void print_usage(char *prog_name) {
    fprintf(stderr, "Usage: \"mpiexec -np <CORES> %s -o <output_file> -m <ROWS> -n <COLS> <input_file> \"\n", prog_name);
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
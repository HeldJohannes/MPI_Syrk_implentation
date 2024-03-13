#include "log.h"
#include "MPI_Syrk_implementation.h"
#include <cblas.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


int read_input(run_config *s, int argc, char* argv[]);
int read_input_file(run_config *s, float *a);

int main(int argc, char* argv[]) {

    int rank, size_of_cluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_of_cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    static run_config config;

    // read the input arguments:
    int return_value = read_input(&config, argc, argv);
    if (return_value != 0) {
        return return_value;
    }

    //define the matrix A:
    float a[config.m * config.n];

    //read the input file:
    return_value = read_input_file(&config, &a);


    MPI_Finalize();
    return EXIT_SUCCESS;
}

int read_input(run_config *s, int argc, char* argv[]) {
    int c;
    char *endptr;
    while ((c = getopt(argc, argv, "m:n:c:")) != -1) {
        switch (c) {
            case 'm':
                s->m = (int) strtol(optarg, &endptr, 0);
                break;
            case 'n':
                s->n = (int) strtol(optarg, &endptr, 0);
                break;
            case 'c':
                s->c = (int) strtol(optarg, &endptr, 0);
            case '?':
                if (optopt == 'm' || optopt == 'n' || optopt == 'c') {
                        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                } else if (isprint(optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return EINVAL;
            default:
                abort();
        }

        if (optind < argc) {
            s->fileName = argv[optind];
        } else {
            fprintf (stderr, "file name is missing");
            return EINVAL;
        }

        return EXIT_SUCCESS;
    }
}

int read_input_file(run_config *s, float *a) {

}

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>

/**
 * Mandatory usage function.
 * @brief This function writes helpful usage information about the program to stderr.
 * @param myprog the program name
 */
void wrong_usage(char *myprog){
    fprintf(stderr, "Usage: %s [-m row_count] [-n col_count] [file...]\n", myprog);
    exit(EXIT_FAILURE);
}

void index_calculation(int *arr, int n, int world_size) {
    int input_size = n / world_size;

    int rest = n % world_size;
    //printf("rest = %d\n", rest);

    for (int i = 0; i < world_size; ++i) {
        arr[i] = input_size;
        if (i < rest) {
            arr[i] += 1;
        }
    }
}

void display(int rank, int len, int array[]) {
    char *string = "rank = %d;";
    printf(string, rank);
    for (int i = 0; i < len; ++i) {
        printf(" %d ", array[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    // Todo input parsing:
    // available options should be -n, -m and -f

    //printf("argc = %d\n", argc);

    if (argc != 6) {
        printf("To many or not enough input variables!\n");
        exit(EXIT_FAILURE);
    }

    //total_row_number
    int m = -1;

    //total_col_number
    int n = -1;

    int opt;
    while ((opt = getopt(argc, argv, "m:n:")) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            default:
            case '?':
                wrong_usage(argv[0]);
        }
    }

    if (m == -1 || n == -1) {
        wrong_usage(argv[0]);
    }

    printf("m = %d; n = %d\n", m, n);



    //input_matrix_in_array_form
    int input[m * n];// = {1, 3, 5,2, 4, 6};

    /*
    input and expected result:
            1,  2               1, 4
            3,  4               2, 5
            5,  6               3, 6
           ------              ------
    1,3,5| 35, 44       1,2,3| 14, 32
    2,4,6| 44, 56       4,5,6| 32, 77

    */

    printf("optind = %d, argc = %d\n", optind, argc);

    char *fileName = argv[optind];
    FILE* stream = fopen(fileName, "r");

    if (stream == NULL) {
        printf("Can't open the file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }



    char line[100];
    int counter = 0;
    while (fgets(line, 100, stream)) {
        char *subtoken, *tmp_line = line;
        for (int i = 0; i < n; ++i) {
            subtoken = strtok(tmp_line, ";");
            tmp_line = strchr(line, ';');
            input[counter] = atoi(subtoken);
            printf("input[%d] = %d\n", counter, input[counter]);
            counter++;
        }
    }

    fclose(stream);




    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int arr[world_size];
    index_calculation(arr, n, world_size);

    //check the size of col that each node gets
    //fprintf(stdout, "rank = %d, size = %d * m\n", rank, arr[rank]);



    //input matrix for each node:
    int rank_input[arr[rank]][m];

    //transposed input matrix for each node:
    int rank_input_t[m][arr[rank]];

    //compute the input matrix and it transpose, 
    // which consists of the columns and all rows in that column:
    int rank_count = rank;
    for (int i = 0; i < rank; ++i) {
        if (i == 0) {
            rank_count = arr[i];
        } else {
            rank_count += arr[i];
        }
        //printf("rank = %d, rank_count = %d\n", rank, rank_count);
    }
    for (int col_count = 0; col_count < arr[rank]; ++col_count) {
        for (int row_count = 0; row_count < m; ++row_count) {
            int tmp = input[(col_count + rank_count) + row_count * n];
            rank_input[col_count][row_count] = tmp;
            rank_input_t[row_count][col_count] = tmp;
            //printf("rank-%d: rank_input[%d][%d] = %d\n", rank, col_count, row_count, rank_input[col_count][row_count]);
            //printf("rank-%d: rank_input_t[%d][%d] = %d\n",rank, row_count, col_count, rank_input_t[row_count][col_count]);
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
            for (int c = 0; c < arr[rank]; ++c) {
                //printf("arr[%d] = %d\n", rank, arr[rank]);
                rank_result[row * m + col] = rank_input[c][row] * rank_input_t[col][c] + rank_result[row * m + col];
                //fprintf(stdout, "rank = %d; i = %d j = %d; result = %d\n", rank, row, col, rank_result[row * m + col]);
                //fprintf(stdout, "rank = %d; i = %d j = %d; result = %d\n", rank, row, col, rank_result[row * m + col]);
            }
        }
    }


    //todo:
    int counts[world_size];
    for (int i = 0; i < world_size; ++i) {
        counts[i] = m;
    }

    int reduction_result[arr[rank] * m];
    for (int i = 0; i < arr[rank] * m; ++i) {
        reduction_result[i] = 0;
    }


    int result = MPI_Reduce_scatter(rank_result, reduction_result, counts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    //todo:
    display(rank, arr[rank] * m, reduction_result);


    MPI_Finalize();

}

#include "one_d_syrk.h"

void syrkIterative(run_config *s, int rank, int index_arr_rank, float **rank_input, float **rank_input_t,
                   float *rank_result) {
    log_trace("[rank %d] syrkIterative()", rank);
    // for each result row:
    for (long row = 0; row < s->m; ++row) {
        // for each result column
        for (long col = 0; col < s->m; ++col) {
            // run for slice of the input:
            for (long c = 0; c < index_arr_rank; ++c) {
                rank_result[row * s->m + col] += rank_input[row][c] * rank_input_t[c][col];
            }
        }
    }
}

void improved_syrkIterative(run_config *s, int rank, const int index_arr_rank, float **rank_input, float **rank_input_t,
                            float *rank_result) {
    log_trace("[rank %d] improved_syrkIterative()", rank);
    for (int row = 0; row < s->m; ++row) {
        //log_debug("outer for loop : row = %d; run_config.m = %d", row, s->m);
        for (int col = row; col < s->m; ++col) {
            //log_debug("middle for loop : col = %d; run_config.n = %d", col, s->m);
            for (int c = 0; c < index_arr_rank; ++c) {

                rank_result[row * s->m + col] += rank_input[row][c] * rank_input_t[c][col];
            }
        }
    }
}

void syrk_withOpenBLAS(run_config *config, int rank, int index_arr_rank, float **rank_input, float *rank_result) {
    log_trace("[rank %d] syrk_withOpenBLAS()", rank);
    // transform 2d array to 1d:
    float *A = (float *) calloc(config->m * index_arr_rank, sizeof(float));
    if (A == NULL) {
        log_error("Calloc failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (int i = 0; i < config->m; ++i) {
        for (int j = 0; j < index_arr_rank; ++j) {
            A[i * index_arr_rank + j] = rank_input[i][j];
        }
    }
    
    // compute syrk:
    cblas_ssyrk64_(
            CblasRowMajor,
            CblasUpper,
            CblasConjNoTrans,
            config->m,
            index_arr_rank,
            1.0f,
            A,
            index_arr_rank,
            0.0f,
            rank_result,
            config->m);
}
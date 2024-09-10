#include "two_d_syrk.h"
#define RANK -1

/**
 * computes (⌊k/c⌋(u − 1) + k) mod c + cu
 * @param k rank of the current processor (0 ≤ k < P)
 * @param u result array position (0 ≤ u < c)
 * @param c prime number such that P=c(c+1)
 */
int func_f(int k, int u, int c) {
    return (k / c * (u - 1) + k) % c + c * u;
}

/**
 * computes h_i(q) = (i − (⌊i/c⌋ − 1) q) mod c + cq
 * @param i (0 ≤ i < c^2)
 * @param q (0 ≤ q < c)
 * @param c prime number such that P=c(c+1)
 * @return the processor assigned bloc
 */
int func_h(int i, int q, int c) {
    return (i - (i / c - 1) * q) % c + c * q;
}

/**
 * specify the set of row block indices that defines the triangle block for a particular processor k
 * @param r_k row block indices array of size c
 * @param k rank of the current processor (0 ≤ k < P)
 */
void calculate_R_k(int *r_k, int k, int c) {
    if (k < c * c) {
        assert(r_k + 0 != NULL);
        r_k[0] = k / c;
        for (int i = 1; i < c; ++i) {
            assert(r_k + i != NULL);
            r_k[i] = func_f(k, i, c);
        }
    } else if (k < (c * c + c)) {
        for (int i = 0; i < c; ++i) {
            assert(r_k + i != NULL);
            r_k[i] = (k - c * c) * c + i;
        }
    }
}

/**
 * d_k defines the diagonal block owned by processor k (|d_k| ≤ 1)
 * @return the index of the diagonal block owned by processor k or -1 if there is none
 */
int calculate_D_k(int k, int c) {
    if (k < c) {
        return -1;
    } else if (k < c * c && k % c == 0) {
        //log_info("[rank == %d] k / c == %d", k, k/c);
        return k / c;
    } else if (k < c * c && k % c != 0) {
        //log_info("[rank == %d] func_f(k, (k / c), c) == %d", k, func_f(k, (k / c), c));
        return func_f(k, (k / c), c);
    }
    return func_f(c * (k - c * c), k - c * c, c);
}

/**
 *
 * @param q_i array of size c+1
 * @param m_bloc_i (0 ≤ i < c^2)
 * @param c prime number such that P=c(c+1)
 */
void calculate_Q_i(int *q_i, int m_bloc_i, int c) {
    if (m_bloc_i < c) {
        for (int i = 0; i < c; ++i) {
            assert(q_i + i != NULL);
            q_i[i] = c * m_bloc_i + i;
        }
        assert(q_i + c != NULL);
        q_i[c] = c * c;
    } else {
        for (int i = 0; i < c; ++i) {
            assert(q_i + i != NULL);
            q_i[i] = func_h(m_bloc_i, i, c);
        }
        assert(q_i + c != NULL);
        q_i[c] = c * c + m_bloc_i / c;
    }
}

int cal_block_size(run_config *s) {
    return (s->m * s->n) / (s->c * s->c * (s->c + 1));
}

void copy_array(const double *src, double *des, int h, int l, int offset_src, int offset_des) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < l; ++j) {
            des[(i * l) + j + offset_des] = src[(i * l) + j + offset_src];
        }
    }
}

void copy_to_2D(float *B, float **A, int k, int block_height, int block_length, int index) {
    int block_size = (block_height * block_length);
    for (int i = 0; i < block_height; ++i) {
        for (int j = 0; j < block_length; ++j) {
            assert(B + (k * block_size + (i * block_length) + j) != NULL);
            assert(&A[i + (block_height * index)][j] != NULL);
            B[k * block_size + (i * block_length) + j] = A[i + (block_height * index)][j];
        }
    }
}

void copy_to_1D(float *A_i, float *B, int r_pos, int w_pos, int block_height, int block_length) {
    int block_size = (block_height * block_length);
    for (int i = 0; i < block_height; ++i) {
        for (int j = 0; j < block_length; ++j) {
            assert(A_i + (w_pos * block_size + (i * block_length) + j) != NULL);
            assert(B + r_pos * block_size + (i * block_length) + j != NULL);
            A_i[w_pos * block_size + (i * block_length) + j] = B[r_pos * block_size + (i * block_length) + j];
        }
    }
}

void copy_to_d(double *A_i, float *A, int c, int block_height, int block_length, int index, _Bool trans) {
    int block_size = (block_height * block_length);
    int shift = index * block_size * (c + 1);
    if (!trans) {
        for (int i = 0; i < block_height; ++i) {
            for (int j = 0; j < c + 1; ++j) {
                for (int k = 0; k < block_length; ++k) {
                    A_i[k + j * block_length + i * block_length * (c + 1)] = A[k + j * block_size + i * block_length +
                                                                               shift];
                }
            }
        }
    } else {
        for (int j = 0; j < c + 1; ++j) {
            for (int k = 0; k < block_length; ++k) {
                for (int i = 0; i < block_height; ++i) {
                    A_i[i + k * block_height + j * block_size] = A[i * block_length + k + j * block_size + shift];
                }
            }
        }
    }
}

void copy_to_f(float *A_i, float *A, int c, int block_height, int block_length, int index, _Bool trans) {
    int block_size = (block_height * block_length);
    int shift = index * block_size * (c + 1);
    if (!trans) {
        for (int i = 0; i < block_height; ++i) {
            for (int j = 0; j < c + 1; ++j) {
                for (int k = 0; k < block_length; ++k) {
                    A_i[k + j * block_length + i * block_length * (c + 1)] = A[k + j * block_size + i * block_length +
                                                                               shift];
                }
            }
        }
    }
}


void cast_d_to_f(double *pDouble, float *pFloat, int size) {
    for (int j = 0; j < size; ++j) {
        pFloat[j] = (float) pDouble[j];
    }
}

/**
 * <p> main function of this class </p>
 *
 * <p> <b>Require:</b> |Π|=P=c(c+1) for prime c </p>
 * <p> <b>Require:</b> A is evenly subdivided into c^2 row blocks, and each row
 *   block A_i is evenly divided across a set of c + 1 processors Q_i </p>
 *
 * @param world_size number of processors
 * @param k rank of the processor [0 ≤ k < world_size]
 * @param m number of rows
 * @param n number of cols
 * @param c prime number such that P=c(c+1)
 */
void two_d_syrk(run_config *s, int k, float *rank_result, float **input) {

    log_trace("rank_result %p", rank_result);

    int block_size = cal_block_size(s);
    int block_height = s->m / (s->c * s->c);
    //log_info("[rank %d] s->m = %d, s->c = %d, block_height = %d", k, s->m ,s->c, block_height);
    int block_length = s->n / (s->c + 1);

    // todo remove:
    // print the input matrix:
    if (k == RANK) {
        for (int l = 0; l < 2; ++l) {
            for (int i = block_height * l; i < block_height * (l + 1); ++i) {
                for (int j = 0; j < block_length; ++j) {
                    printf("%0.0f ", input[i][j]);
                }
            }
            printf("\n");
        }
    }


    int *R_k = (int *) calloc(s->c, sizeof(int));
    calculate_R_k(R_k, k, s->c);

    int *Q_i = (int *) calloc(s->c + 1, sizeof(int));

    // Gather c row blocks in row block set:

    // Step 1:
    // allocate array B of P blocks, each of size [m*n / c^2 (c+1)]
    float *B = (float *) calloc(s->world_size * block_size, sizeof(float));
    // error checking
    if (B == NULL) {
        log_fatal("Memory allocation failed for input");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Step 2:
    // copy the input matrix block A^(k) to position B_k\'
    for (int i = 0; i < s->c; ++i) {
        // for each i ∈ R_k do:
        calculate_Q_i(Q_i, R_k[i], s->c);
        for (int j = 0; j < s->c + 1; ++j) {
            // for each k ∈ Q_i \{k} do:
            if (Q_i[j] != k) {
                // copy the bloc A^(k) to B_k
                copy_to_2D(B, input, Q_i[j], block_height, block_length, i);
            }
        }
    }


    // todo remove:
    // print the matrix B after copy:
    if (k == RANK) {
        log_info("TEST, world_size=%d, block_height=%d, block_length=%d , %d, %d", s->world_size, block_height,
                 block_length, s->world_size * block_height, s->world_size * block_length);
        for (int i = 0; i < s->world_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                assert(&B[i * block_size + j] != NULL);
                printf("%0.0f ", B[i * block_size + j]);
            }
            printf("\t");
        }
        printf("\n");
    }

    // Step 3:
    // Communicate B ALL-TO-ALL
    MPI_Alltoall(B, block_size, MPI_FLOAT, B, block_size, MPI_FLOAT, MPI_COMM_WORLD);


    // todo remove:
    // print the matrix B after ALLtoALL:
    if (k == RANK) {
        log_info("TEST, world_size=%d, block_height=%d, block_length=%d , %d, %d", s->world_size, block_height,
                 block_length, s->world_size * block_height, s->world_size * block_length);
        for (int i = 0; i < s->world_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                assert(&B[i * block_size + j] != NULL);
                printf("%0.0f ", B[i * block_size + j]);
            }
            printf("\t");
        }
        printf("\n");
    }

    float *A = (float *) calloc((s->c + 1) * 2 * block_size, sizeof(float));

    // Step 4:
    // Accumulate B_k′ into A
    for (int i = 0; i < s->c; ++i) {
        // for each i ∈ R_k do:
        calculate_Q_i(Q_i, R_k[i], s->c);
        for (int j = 0; j < s->c + 1; ++j) {
            // for each k ∈ Q_i do:
            // Accumulate B_k′ into A
            if (Q_i[j] != k) {
                copy_to_1D(A, B, Q_i[j], j + ((s->c + 1) * i), block_height, block_length);
            } else {
                copy_to_2D(A, input,j+ (i * (s->c + 1)), block_height, block_length, i);
            }
        }
    }


    //todo remove:
    // print A:
    if (k == RANK) {
        log_info("A:");
        for (int i = 0; i < (s->c + 1) * 2; ++i) {
            for (int j = 0; j < block_size; ++j) {
                printf("%0.0f ", A[i * block_size + j]);
            }
            printf("\t");
        }
    }
    //*/

    // Step 5:
    // Compute c(c − 1)/2 off-diagonal blocks

    // for each (i,j) ∈ R_k with i > j do:
    double *result = (double *) calloc(block_height * block_height, sizeof(double));
    float *result_f = (float *) calloc(block_height * block_height, sizeof(float));
    double *A_i = (double *) calloc(block_height * s->n, sizeof(double));
    double *A_j = (double *) calloc(block_height * s->n, sizeof(double));
    for (int i = 0; i < s->c; ++i) {
        for (int j = 0; j < s->c; ++j) {
            if (R_k[i] > R_k[j]) {
                copy_to_d(A_i, A, s->c, block_height, block_length, i, false);
                if (k == RANK) {
                    log_info("A_i:");
                    for (int l = 0; l < block_height * s->n; ++l) {
                        printf("%0.0f ", A_i[l]);
                    }
                    printf("\n");
                }
                copy_to_d(A_j, A, s->c, block_height, block_length, j, true);
                if (k == RANK) {
                    log_info("A_j:");
                    for (int l = 0; l < block_height * s->n; ++l) {
                        printf("%0.0f ", A_j[l]);
                    }
                    printf("\n");
                }
                //C_ij =Local-GEMM(A,A_j^T)
                //gemm((const double *) &A[i * block_size], (const double *) &A[j * block_size], block_height, block_length, result);
                cblas_dgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        block_height, block_height, s->n,
                        1.0,
                        A_i,
                        s->n,
                        A_j,
                        block_height,
                        0.0,
                        result,
                        block_height);
                // write result to C_ij
                cast_d_to_f(result, result_f, block_height * block_height);
                copy_to_1D(rank_result, result_f, 0, R_k[i] * (block_height) + R_k[j], block_height,
                           block_height);
            }
        }
    }
    //todo remove
    // print result
    if (k == RANK) {
        //
        //       [,1]  [,2]  [,3]  [,4]
        //  [1,] 7562 24650 13787 11729
        //  [2,] 9614 26208 12463 13103
        //  [3,] 5337 12039  5829  7168
        //  [4,] 6585 20325  9343  9088
        //
        log_info("result:");
        for (int i = 0; i < block_height; ++i) {
            for (int j = 0; j < block_height; ++j) {
                printf("%0.0f ", result[i * block_height + j]);
            }
            printf("\n");
        }
    }
    free(A_i);
    free(result_f);
    free(result);
    free(A_j);

    // Step 6
    // Compute diagonal block if assigned

    // for each i ∈ D_k do
    int d_k = calculate_D_k(k, s->c);
    float *result_D_k = (float *) calloc(block_height * block_height, sizeof(float ));
    float *A_i_D_k = (float *) calloc(block_height * s->n, sizeof(float ));
    if (d_k != -1) {
        for (int i = 0; i < s->c; ++i) {
            if (R_k[i] == d_k) {
                //log_info("[rank == %d] i == %d", k, i);
                copy_to_f(A_i_D_k, A, s->c, block_height, block_length, i, false);
                //C_ii = Local-SYRK(A_i)
                cblas_ssyrk(CblasRowMajor,CblasLower,CblasConjNoTrans,
                            block_height,s->n,
                            1.0f,A_i_D_k,s->n,
                            0.0f,result_D_k,block_height);
                copy_to_1D(rank_result, result_D_k, 0, d_k * (block_height) + d_k, block_height,block_height);
            }
        }
    }
    free(result_D_k);
    free(A_i_D_k);

    //todo remove
    // print rank_result
    if (k == RANK) {
        log_info("rank_result:");
        for (int i = 0; i < s->m; ++i) {
            for (int j = 0; j < block_height; ++j) {
                for (int l = 0; l < block_height; ++l) {
                    printf("%0.0f ", rank_result[i * s->m + j * block_height + l]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }



    // cleanup
    free(R_k);
    free(Q_i);
    free(B);
    free(A);
}
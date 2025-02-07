//
// Created by Johannes Held on 10.07.24.
//

#ifndef MPI_SYRK_IMPLEMENTATION_TWO_D_SYRK_H
#define MPI_SYRK_IMPLEMENTATION_TWO_D_SYRK_H

#include "MPI_Syrk_implementation.h"

void calculate_Q_i(int *q_i, int m_bloc_i, int c);

int cal_block_size(run_config *s);

void two_d_syrk(run_config *s, int k, float *rank_result, float **input);

#endif //MPI_SYRK_IMPLEMENTATION_TWO_D_SYRK_H

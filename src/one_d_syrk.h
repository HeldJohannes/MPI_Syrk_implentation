#ifndef MPI_SYRK_IMPLEMENTATION_ONE_D_SYRK_H
#define MPI_SYRK_IMPLEMENTATION_ONE_D_SYRK_H

#include "MPI_Syrk_implementation.h"

extern void syrkIterative(run_config *config, int rank, int index_arr, float** rank_input, float** rank_input_t, float* rank_result);

extern void improved_syrkIterative(run_config *config, int rank, int index_arr, float** rank_input,
                            float** rank_input_t, float* rank_result);

extern void syrk_withOpenBLAS(run_config *config, int rank, int index_arr, float** rank_input, float* rank_result);

#endif //MPI_SYRK_IMPLEMENTATION_TWO_D_SYRK_H

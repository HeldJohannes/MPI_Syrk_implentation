set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)

set(SOURCES main.c log.c
        two_d_syrk.c
        two_d_syrk.h
        one_d_syrk.c)
add_executable(MPI_SYRK_implementation ${SOURCES})


# Add libraries for code completion and compiling
target_link_libraries(MPI_SYRK_implementation PUBLIC MPI::MPI_CXX BLAS::BLAS)
target_compile_options(MPI_SYRK_implementation PRIVATE -Werror -Wall -Wextra)

message(STATUS "Run: \" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS \" ")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)

set(SOURCES 
        MPI_Syrk_implementation.c
        log.c
        two_d_syrk.c
        one_d_syrk.c
)

set(HEADERS
        MPI_Syrk_implementation.h
        log.h
        two_d_syrk.h
        one_d_syrk.h
)

add_library(MPI_SYRK_lib STATIC ${SOURCES} ${HEADERS})
target_link_libraries(MPI_SYRK_lib PUBLIC MPI::MPI_CXX BLAS::BLAS)

# Add executable
add_executable(MPI_SYRK_implementation main.c)


# Add libraries for code completion and compiling
target_link_libraries(MPI_SYRK_implementation PUBLIC MPI_SYRK_lib MPI::MPI_CXX BLAS::BLAS)
target_compile_options(MPI_SYRK_implementation PRIVATE -Werror -Wall -Wextra)

message(STATUS "Run: \" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS \" ")

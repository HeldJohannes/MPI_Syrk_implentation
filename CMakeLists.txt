########################################################################
# CMake file for the project MPI_Syrk_implementation
#
########################################################################

cmake_minimum_required(VERSION 3.9...3.29)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()



project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes")


# Checks if MPI is available on this machine
find_package(MPI REQUIRED)
message("Found MPI LIBARIES: " ${MPI_FOUND})
if (!MPI_FOUND)
    message(FATAL_ERROR "MPI not found")
endif (!MPI_FOUND)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)


#set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/spack/spack_git_updated/opt/spack/linux-debian11-skylake_avx512/gcc-12.1.0/openblas-0.3.26-qp6eunppchtpxplzta3jd5komgb2nn6r")
#message(CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}")


#set(BLA_VENDOR OpenBLAS)
#if (APPLE)
#	message("IS APPLE TRUE")
#    set(CMAKE_PREFIX_PATH  "/opt/homebrew/Cellar/openblas/0.3.27/")
#    include_directories("/opt/homebrew/Cellar/openblas/0.3.27/include/")
#endif (APPLE)
#find_package(BLAS)


#link openBlas

#message("Found BLAS LIBARIES: " ${BLAS_FOUND})
#if (BLAS_FOUND)
#    message("Found the BLAS LIBARIE: " ${BLAS_LIBRARIES})
#endif (BLAS_FOUND)

#include_directories("/opt/homebrew/include/")

set(SOURCES src/main.c src/log.c src/MPI_Syrk_implementation.h)

add_executable(MPI_SYRK_implementation ${SOURCES})


# Add libraries for code completion and compiling
target_link_libraries(MPI_SYRK_implementation PUBLIC MPI::MPI_CXX)

message(STATUS "Run: \" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS \" ")

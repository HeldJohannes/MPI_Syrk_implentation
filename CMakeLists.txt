########################################################################
# CMake file for the project MPI_Syrk_implementation
#
########################################################################

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()



project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes")

# Checks if MPI is available on this machine
find_package(MPI REQUIRED)

set(BLA_VENDOR OpenBLAS)
set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/openblas/0.3.24")
if (APPLE)
    include_directories("/opt/homebrew/Cellar/openblas/0.3.24/include")
endif (APPLE)
find_package(BLAS REQUIRED)

find_path(BLAS_INCLUDE_DIRS cblas.h
        $ENV{BLAS_HOME}/include)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)

#link openBlas

message("Found BLAS LIBARIES: " ${BLAS_FOUND})
if (BLAS_FOUND)
    message("Found OPEN_BLAS LIBARIES: " ${BLAS_LIBRARIES})
endif (BLAS_FOUND)

set(SOURCES main.c log.c MPI_Syrk_implementation.h)

add_executable(MPI_SYRK_implementation ${SOURCES})


# Add libraries for code completion and compiling
target_link_libraries(MPI_SYRK_implementation PUBLIC MPI::MPI_CXX BLAS::BLAS)

message(STATUS "Run: \" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS \" ")

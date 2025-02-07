########################################################################
# CMake file for the project MPI_Syrk_implementation
#
########################################################################

cmake_minimum_required(VERSION 3.18)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes")

set(HYDRA True)

# Checks if MPI is available on this machine
if (HYDRA)
    list(APPEND CMAKE_PREFIX_PATH "/opt/mpi/openmpi-4.1.4")
endif(HYDRA)
find_package(MPI REQUIRED)
message("Found MPI LIBARIES: " ${MPI_FOUND} " -> " ${MPI_C_LIBRARIES})
message("Added MPI header dir: " ${MPI_C_INCLUDE_DIRS})
if (!MPI_FOUND)
    message(FATAL_ERROR "MPI not found")
endif (!MPI_FOUND)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)

# link openBlas
set(BLA_VENDOR OpenBLAS)
if(HYDRA)
    set(ENV{PKG_CONFIG_PATH} "/home/thesis/jheld/MPI_Syrk_implementation")
    message("PKG_CONFIG_PATH = " ${PKG_CONFIG_PATH})
    set(BLA_PREFER_PKGCONFIG TRUE)
endif(HYDRA)
if (APPLE)
    include_directories(/opt/homebrew/opt/openblas/lib)
    set(BLA_INCLUDE_DIRS /opt/homebrew/include)
endif(APPLE)

find_package(BLAS REQUIRED GLOBAL)
message("Found BLAS LIBARIES: " ${BLAS_FOUND} " -> " ${BLAS_LIBRARIES})
message("Found BLAS INCLUDE header files: " ${BLA_INCLUDE_DIRS})
include_directories(${BLA_INCLUDE_DIRS})
if (HYDRA) 
    message("HYDRA")
    include_directories(/opt/spack/spack_git_updated/opt/spack/linux-debian11-skylake_avx512/gcc-12.1.0/openblas-0.3.26-qp6eunppchtpxplzta3jd5komgb2nn6r/include)
endif(HYDRA)
message("Added BLAS header dir: " ${BLA_INCLUDE_DIRS})

include_directories(src)

add_subdirectory(src)
find_package(GTest)
if (GTEST_FOUND)
    add_subdirectory(test)
endif(GTEST_FOUND)
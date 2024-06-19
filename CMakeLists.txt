########################################################################
# CMake file for the project MPI_Syrk_implementation
#
########################################################################

cmake_minimum_required(VERSION 3.28)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()



project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)

# Checks if MPI is available on this machine
find_package(MPI 3.1 REQUIRED)
message("Found MPI LIBARIES: " ${MPI_FOUND} " -> " ${MPI_C_LIBRARIES})
message("Added MPI header dir: " ${MPI_C_INCLUDE_DIRS})
if (!MPI_FOUND)
    message(FATAL_ERROR "MPI not found")
endif (!MPI_FOUND)

# link openBlas
set(BLA_VENDOR OpenBLAS)
include_directories(/opt/homebrew/opt/openblas/lib)
set(BLA_INCLUDE_DIRS /opt/homebrew/include)
find_package(BLAS REQUIRED GLOBAL)
message("Found BLAS LIBARIES: " ${BLAS_FOUND} " -> " ${BLAS_LIBRARIES})
include_directories(${BLA_INCLUDE_DIRS})
message("Added BLAS header dir: " ${BLA_INCLUDE_DIRS})
message("")

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
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
message("Found MPI LIBARIES: " ${MPI_FOUND} " -> " ${MPI_C_LIBRARIES})
message("Added MPI header dir: " ${MPI_C_INCLUDE_DIRS})
if (!MPI_FOUND)
    message(FATAL_ERROR "MPI not found")
endif (!MPI_FOUND)

#link openBlas
set(BLA_VENDOR OpenBLAS)
set(BLA_INCLUDE_DIRS /opt/homebrew/Cellar/openblas/0.3.27/include)
find_package(BLAS REQUIRED GLOBAL)
message("Found BLAS LIBARIES: " ${BLAS_FOUND} " -> " ${BLAS_LIBRARIES})
include_directories(${BLA_INCLUDE_DIRS})
message("Added BLAS header dir: " ${BLA_INCLUDE_DIRS})
message("")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicc)





set(SOURCES src/main.c src/log.c src/MPI_Syrk_implementation.h)

add_executable(MPI_SYRK_implementation ${SOURCES})


# Add libraries for code completion and compiling
target_link_libraries(MPI_SYRK_implementation PUBLIC MPI::MPI_CXX BLAS::BLAS)
target_compile_options(MPI_SYRK_implementation PRIVATE -Werror -Wall -Wextra)

message(STATUS "Run: \" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS \" ")

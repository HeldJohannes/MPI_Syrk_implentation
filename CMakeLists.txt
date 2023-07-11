cmake_minimum_required(VERSION 3.9...3.29)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

SET(CMAKE_C_COMPILER gcc)

project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes")

find_package(MPI REQUIRED)

add_executable(MPI_SYRK_implementation main.c log.c)

message(STATUS "Run ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_SYRK_implementation ${MPIEXEC_POSTFLAGS} ARGS")

target_link_libraries(MPI_SYRK_implementation PUBLIC MPI::MPI_CXX)


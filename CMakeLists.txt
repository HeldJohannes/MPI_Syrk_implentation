########################################################################
# CMake file for the project MPI_Syrk_implementation
#
########################################################################

cmake_minimum_required(VERSION 3.18)

option(HYDRA "Working on HYDRA" ON)
option(PACKAGE_TESTS "Build the tests" ON)

project(MPI_SYRK_implementation
        VERSION 1.0
        DESCRIPTION "Implementation for the bachelor thesis by Held Johannes"
)

# MPI & BLAS Konfigurationen
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)
include(MPIConfig)
include(BLASConfig)

# Quellcode-Verzeichnis einbinden
include_directories(src)
add_subdirectory(src)

if (PACKAGE_TESTS)
    add_subdirectory(test)
endif()
# MPI finden und konfigurieren
if (HYDRA)
    list(APPEND CMAKE_PREFIX_PATH "/opt/mpi/openmpi-4.1.4")
endif()

find_package(MPI REQUIRED)

if (MPI_FOUND)
    message(STATUS "MPI gefunden: ${MPI_C_LIBRARIES}")
    message(STATUS "MPI Include-Pfade: ${MPI_C_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "MPI nicht gefunden")
endif()

# Includes setzen
include_directories(${MPI_C_INCLUDE_DIRS})

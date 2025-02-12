# OpenBLAS-Suche
set(BLA_VENDOR OpenBLAS)
set(BLA_SIZEOF_INTEGER 8)

if (HYDRA)
    set(ENV{PKG_CONFIG_PATH} "/home/thesis/jheld/MPI_Syrk_implementation")
    message(STATUS "PKG_CONFIG_PATH = $ENV{PKG_CONFIG_PATH}")
    set(BLA_PREFER_PKGCONFIG TRUE)
endif()

if (APPLE)
    include_directories(/opt/homebrew/opt/openblas/lib)
    set(BLA_INCLUDE_DIRS /opt/homebrew/include)
endif()

find_package(BLAS REQUIRED)

if (BLAS_FOUND)
    message(STATUS "BLAS gefunden: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS Include-Pfad: ${BLA_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "BLAS nicht gefunden")
endif()

include_directories(${BLA_INCLUDE_DIRS})

if (HYDRA)
    include_directories(
        /opt/spack/spack_git_updated/opt/spack/linux-debian11-skylake_avx512/gcc-12.1.0/openblas-0.3.26-qp6eunppchtpxplzta3jd5komgb2nn6r/include
    )
endif()

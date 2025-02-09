# MPI_Syrk_implentation

## Overview

Implementation of the BLAS level 3 algorithm SYRK (Symmetric -k Rank update) using MPI.

## Prerequisite

The following software is needed to run this Project:

1. Git
2. Git LFS
3. OpenBlas
4. OpenMPI

   <details>
        <summary>Under Mac OS</summary>

    - ### Using `brew`
        - Install

          to install OpenMPI with brew  you can use the following command
          `brew install open-mpi`
    - Commands
        - mpicc
        - mpirun

   </details>

## Getting Started

Download the repository with:

```
git clone HeldJohannes/MPI_Syrk_implementation
```

The test files are to big for normal git, for this reason the repository uses git LFS.
To receive the files `git pull` should automatically pull the files if it isn't the case run:

```
git lfs pull
```

To compile run

```
make
```

## Usage

```
mpirun -np <CORES> MPI_SYRK_implementation -m <ROWS> -n <COLS> -a <ALGORITHM> <input_file>
```

Testfiles and expected result files can be found in the folder **./resource/input** and **./resource/in**.

In the folder **./resource/output** are the expected output for the testcases to check the test of **./resource/input**

The first number is the number of rows and the second the number of columns.

## Example

input and expected result:

```
       
        1,  2                      1,  4
        3,  4                      2,  5
        5,  6                      3,  6
       ------                    ------
1,3,5| 35, 44              1,2,3| 14, 32
2,4,6| 44, 56              4,5,6| 32, 77
```
## Check if the computation is correct 

You can use the R script `./resource/Check.R` to check if the computation is correct

```
Rscript --vanilla ./resource/Check.R <result_file> <input_file> [TRUE|FALSE]
```

where `result_file` the file is which was the output of `MPI_SYRK_implementation`
and `<input_file>` is the file for which `MPI_SYRK_implementation` computed the result
`[TRUE|FALSE]` tell the script to ignore the lower half of the result matrix.

### Example

```
mpirun -np 1 ./cmake-build-debug/src/MPI_SYRK_implementation -m 60 -n 40 -a 0 ./resource/input/test60x40.csv;
Rscript --vanilla ./resource/Check.R ./syrk_result.csv ./resource/input/test60x40.csv FALSE
```

## Generate Testdata

You can use the R script in the test folder to generate testdata:

```
Rscript --vanilla ./resourec/GernateTestdata.R <dir_to_safe> <Number of Rows (m)> <Number of Cols (n)>
```

## Helpful Links

- [Open MPI](https://www.jetbrains.com/help/clion/openmpi.html)
- [Pkg-Config](https://people.freedesktop.org/~dbn/pkg-config-guide.html)
- [C Make FindBLAS](https://cmake.org/cmake/help/v3.18/module/FindBLAS.html)
- [C Make find_package()](https://cmake.org/cmake/help/v3.18/command/find_package.html#search-modes)
- [Slurm tutorial](https://www.uibk.ac.at/zid/systeme/hpc-systeme/common/tutorials/slurm-tutorial.html#HDR2_1_1)


## Helpful commads on server

The module mpi has to be loaded every day with:
```
module load mpi/openmpiS
```
The loaded modeles can be checked with:
```
module list
```
To get the current directory path:
```
pwd
```
To add the `blas.pc`to the pkg-config path use the following command in the directory where the file is:
```
export PKG_CONFIG_PATH=$(pwd)
```
to check that the command worked use:
```
echo $PKG_CONFIG_PATH
```
to check that the pkg-config is found by the pkg-config command use:
```
pkg-config --modversion blas
```
to build the system on `HYDRA`:
```
mkdir build
cd build/
cmake .. -DHYDRA=True
cmake --build .
```

## alternativ:

build the config files 
```
cmake -S . -B build -DHYDRA
```
build the code
```
cmake --build build
```

to test that everything works as expected:
```
cd /home/thesis/jheld/MPI_Syrk_implementation
mpiexec -np 1 ./build/src/MPI_SYRK_implementation -m 2 -n 3 -a 2 ./resource/input/test2x3.csv
```
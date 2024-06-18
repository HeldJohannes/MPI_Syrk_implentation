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
Rscript --vanilla ./test/GernateTestdata.R <dir_to_safe> <Number of Rows (m)> <Number of Cols (n)>
```

## Helpful Links

- https://www.jetbrains.com/help/clion/openmpi.html
- 
# MPI_Syrk_implentation

## Overview

Implementation of the BLAS level 3 algorithm SYRK (Symmetric -k Rank update) using MPI.

## Usage

```
make
mpirun -np <CORES> MPI_SYRK_implementation -m <ROWS> -n <COLS> <input_file>
```

Testfiles and expected result files can be found in the folder **/test**.
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

## Getting Started

1. Install OpenMPI

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

## Generate Testdata

You can use the R script in the test folder to generate testdata:

```
Rscript --vanilla ./test/GernateTestdata.R <dir_to_safe> <Number of Rows (m)> <Number of Cols (n)>
```

## Helpful Links

- https://www.jetbrains.com/help/clion/openmpi.html
- 
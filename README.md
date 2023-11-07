# MPI_Syrk_implementation

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

## Definitions:

SYRK := $$C <- \alpha * A * A^T + \beta * C$$
or      $C <- \alpha * A^T * A + \beta * C$

Eingabe Matrix $A$ hat die dimensionen $A_{m x n}$

m -> number of rows in the matrix $A$
n -> number of COL in der matrix $A$

Bei einer Matrizenmultiplikation muss die Spaltenzahl der ersten Matrix gleich der Zeilenzahl der zweiten Matrix sein. 
Die Ergebnismatrix hat dann die Zeilenzahl der ersten und die Spaltenzahl der zweiten Matrix.

Ergebnis Matrix C hat die dimensionen $C_{m x m}$

Aus $A_{m x n}$ folgt $A^T = A_{n x m}$
Daraus folgt $A_{m x n} * A^T = C_{m x m}$
## Getting Started

1. Install OpenMPI

   <details>
        <summary>Under Mac OS</summary>
   
   - ### Using `brew`
     - Install
   
          to install OpenMPI with brew  you can use the following command 
          `brew install open-mpi`
   - Commands
     - `mpicc`
     - `mpirun`
      
   </details>

## Helpful Links

- https://www.jetbrains.com/help/clion/openmpi.html
- 
# MPI_Syrk_implentation

## Usage

```
make
mpirun -np ... MPI_SYRK_implementation -m ... -n ... <input_file>
```

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
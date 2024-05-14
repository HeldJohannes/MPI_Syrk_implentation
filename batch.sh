#!/bin/bash

# Program to run (replace with your actual program name)
program_name="mpirun"

executionFile="/Users/johannes/CLionProjects/MPI_Syrk_implentation/cmake-build-debug/MPI_SYRK_implementation"

# Folder containing the files (replace with your actual folder path)
folder_path="/Users/johannes/CLionProjects/MPI_Syrk_implentation/test/in"

# Output file for collected results (replace with your desired filename)
output_file="results.csv"

number_of_runs=3

# Clear the output file (optional)
echo "" > "$output_file"

# Header row for the CSV (modify column names if needed)
echo "number_of_processors;matrix_size_m;matrix_size_n;run_number;run_time" > "$output_file"


x=1
for (( i = 1; i <= x; i++ )); do

echo "number of processors = $i"

# Loop through each file in the folder
  for file in "${folder_path}"/*; do
    # Check if it's a regular file (skip hidden files, etc.)
    if [[ -f "$file" ]]; then
      # Extract matrix dimensions from filename
      m=$(sed -E 's/^.*test([0-9]+)x([0-9]+).*$/\1/' <<< "$file")
      n=$(sed -E 's/^.*test([0-9]+)x([0-9]+).*$/\2/' <<< "$file")


      for (( j = 0; j < number_of_runs; j++ )); do
        # Build the command string
        command="$program_name -np $i $executionFile -m $m -n $n $file"
        echo "$command"

        "$program_name" -np "$i" "$executionFile" -m "$m" -n "$n" "$file" > /tmp/script_output.txt

        # Execute the command and capture runtime
        run_time=$(grep -Eo '[0-9]+\.[0-9]+' <<< cat /tmp/script_output.txt)

        # Print data in CSV format (modify column order if needed)
        echo "$i;$m;$n;$((j + 1));$run_time" >> "$output_file"
      done
    fi
  done
done


# Print confirmation message
echo "Results collected in: $output_file"

###############################################################################
#
# Generator Script for Test-Data
#
###############################################################################

options(scipen = +10)

# Function to generate and save a random integer matrix
generate_and_save_matrix <- function(m, n, filename) {
  
  # Check if both m and n are numeric
  if (!is.numeric(m) || !is.numeric(n)) {
    stop("Error: Both m and n must be numeric values")
  }
  
  # Generate random matrix with values between 0 and 100 (inclusive)
  random_matrix <- runif(m * n, min = 0, max = 100)
  
  # Reshape the vector into a matrix with specified dimensions
  random_matrix <- matrix(random_matrix, nrow = m, ncol = n)
  
  # Round the values in the matrix to integers
  rounded_matrix <- round(random_matrix)
  
  # Write the matrix to a CSV file
  write.table(rounded_matrix, file = filename, row.names = F, col.names = FALSE, sep = ";")
}

# Get input arguments from the console
args <- commandArgs(trailingOnly = TRUE)

# Check if required arguments are provided
if (length(args) < 2) {
  stop("Error: Missing required arguments. Usage: GenerateTestdata.R <output_file> <m> <n>")
}

# Extract file path, m, and n from arguments
dir <- args[[1]]
n_rows<- as.numeric(args[[2]])
n_cols <- as.numeric(args[[3]])

# Define matrix size and filename
#n_rows <- 1e+01
#n_cols <- 1e+01
#dir <- "/Users/johannes/CLionProjects/MPI_Syrk_implentation/test/in/"
filename <- paste(dir, "test", n_rows, "x", n_cols, ".csv", sep = "")

# Generate and save the random matrix
generate_and_save_matrix(n_rows, n_cols, filename)

print(paste("Random matrix saved to:", filename))


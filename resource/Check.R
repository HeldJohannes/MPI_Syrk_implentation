# Get input arguments from the console
args <- commandArgs(trailingOnly = TRUE)

# Check if required arguments are provided
if (length(args) < 3) {
  stop("Error: Missing required arguments. Usage: Check.R <result_file> <input_file> [TRUE|FALSE]")
}

# Assign input arguments to variables
filename_result <- args[[1]]
filename_input <- args[[2]]
ignore_lower_half <- tolower(args[[3]]) == "true"

# Read the result data CSV
result_data <- read.csv(filename_result, sep = ";", header = FALSE)

# Read the expected data CSV
input_data <- as.matrix(read.csv(filename_input, sep = ";", header = FALSE))

# Check if the dimensions of the result data and the computed SYRK are compatible
if (ncol(result_data) != nrow(result_data)) {
  stop("Error: Result data must be a square matrix.")
}

# Compute SYRK for the input file
computed_data <- input_data %*% t(input_data)

# Optionally ignore the lower half of the computed matrix
if (ignore_lower_half) {
  message("Ignoring lower half of the computed matrix")
  computed_data[lower.tri(computed_data)] <- 0
}

# Check if the result data and computed data are identical
comparison_result <- all(result_data == computed_data)
message(paste("The files are equal:", comparison_result))


################################################################################
# Setup:
################################################################################

# add necessary libraries:
library(dplyr)
library(tidyr)
library(ggplot2)

# set the directory to read form and the filename
dir <- "/Users/johannes/CLionProjects/MPI_Syrk_implentation/"
filename <- "results.csv"

# read the result data CSV
measurment_data <- read.csv(paste(dir, filename, sep = ""), sep = ";")

################################################################################
# Evaluation:
################################################################################

# group the date and summarise the runtime over all runs and 
# unite the matrix size to a string for better readability
avg_data <- measurment_data %>%
  arrange(algorithm, number_of_processors, matrix_size_m, matrix_size_n) %>%
  group_by(algorithm, number_of_processors, matrix_size_m, matrix_size_n) %>%
  summarise(avg_runtime = mean(run_time)) %>%
  unite(matrix_shape, c(matrix_size_m, matrix_size_n), sep = "x")

# plot the data using ggplot:
avg_data %>%
  ggplot(aes(x = matrix_shape, y = avg_runtime, colour = factor(number_of_processors))) +
  geom_point() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1))


# filter for matrix 10x... 
avg_data_2 <- measurment_data %>%
  filter(matrix_size_m <= 10) %>%
  arrange(algorithm, number_of_processors, matrix_size_m, matrix_size_n) %>%
  group_by(algorithm, number_of_processors, matrix_size_m, matrix_size_n) %>%
  summarise(avg_runtime = mean(run_time)) %>%
  unite(matrix_shape, c(matrix_size_m, matrix_size_n), sep = "x")

# plot filtered data:
avg_data_2 %>%
  ggplot(aes(x = matrix_shape, y = avg_runtime, colour = factor(number_of_processors))) +
  geom_point() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

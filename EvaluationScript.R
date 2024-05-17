dir <- "/Users/johannes/CLionProjects/MPI_Syrk_implentation/"
filename <- "results.csv"
measurment_data <- read.csv(paste(dir, filename, sep = ""), sep = ";")

avg_data <- measurment_data %>%
  arrange(number_of_processors, matrix_size_m, matrix_size_n) %>%
  group_by(number_of_processors, matrix_size_m, matrix_size_n) %>%
  summarise(avg_runtime = mean(run_time)) %>%
  unite(matrix_shape, c(matrix_size_m, matrix_size_n), sep = "x")

avg_data %>%
  ggplot(aes(x = matrix_shape, y = avg_runtime, colour = factor(number_of_processors))) +
  geom_point() +
  scale()
theme(axis.text.x = element_text(angle = 60, hjust = 1))
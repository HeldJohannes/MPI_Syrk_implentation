# gTest
find_package(GTest CONFIG REQUIRED)
include_directories(/opt/homebrew/Cellar/googletest/1.14.0/include/)
add_executable(MPI_SYRK_test test.cpp)
target_link_libraries(MPI_SYRK_test gtest gtest_main)
add_test(NAME example_test COMMAND MPI_SYRK_test)
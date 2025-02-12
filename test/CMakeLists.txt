cmake_minimum_required(VERSION 3.18)


# Enforce C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing framework
enable_testing()

set(SOURCES 
    test.cpp
    test_parseInput.cpp
)

# Fetch GoogleTest via CMake FetchContent
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.16.0
)
FetchContent_GetProperties(googletest)

# Method to fetch and set up GoogleTest
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(MPI_SYRK_test ${SOURCES})

# Link against GoogleTest
target_link_libraries(MPI_SYRK_test PRIVATE MPI_SYRK_lib GTest::gmock GTest::gtest_main)

# Auto-discover tests
include(GoogleTest)
gtest_discover_tests(MPI_SYRK_test)
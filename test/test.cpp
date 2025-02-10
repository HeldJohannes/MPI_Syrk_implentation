#include <gtest/gtest.h>
#include <gtest/gtest-matchers.h>
#include <gmock/gmock-matchers.h>
#include <vector>
#include "MPI_Syrk_implementation.h"

using ::testing::ElementsAreArray;

class DateConverterFixture : public ::testing::Test {

protected:

    int m = 2;
    int n = 2;

    float** arr_input;
    float** result;


    virtual void SetUp(){

        // Allocate memory for float**
        arr_input = new float*[m];
        result = new float*[n];

        for (int i = 0; i < m; ++i) {
            arr_input[i] = new float[n];
        }
        for (int i = 0; i < n; ++i) {
            result[i] = new float[m];  // Transposed matrix will be n x m
        }

        // Fill arr_input with test data
        int counter = 1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                arr_input[i][j] = static_cast<float>(counter++);
            }
        }
    }

    virtual void TearDown() {

    }


};

TEST_F(DateConverterFixture, transposeMatrix_should_work) {

    transposeMatrix(m, n, arr_input, result);
    
    // Flatten `result` matrix into a 1D vector for comparison
    std::vector<float> result_flat;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result_flat.push_back(result[i][j]);
        }
    }

    // Expected transposed matrix
    std::vector<float> expected = {1, 3, 2, 4};

    EXPECT_THAT(result_flat, ElementsAreArray(expected));
}

// Test if the gtest framwork is correctly integrated with some basic assertions.
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
  }
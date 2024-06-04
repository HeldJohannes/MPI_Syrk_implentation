#include <gtest/gtest.h>
#include <gtest/gtest-matchers.h>
#include <gmock/gmock-matchers.h>
#include "MPI_Syrk_implementation.h"

class DateConverterFixture : public ::testing::Test {

protected:
    virtual void SetUp()
    {

    }

    virtual void TearDown() {

    }


};

TEST_F(DateConverterFixture, transposeMatrix_should_work) {

    int m = 2;
    int n = 2;
    float *arr_input = (float *) calloc(4, sizeof(float ));
    float *result = (float *) calloc(4, sizeof(float));

    for (int i = 0; i < m*n; ++i) {
        arr_input[i] = (float) i + 1.0f;
    }

    transposeMatrix(m, n, &arr_input, result);
    ASSERT_THAT(result, ElementsAreArray({1, 3, 2, 4}));

    free(arr_input);
    free(result);
}
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include "MPI_Syrk_implementation.h"

using ::testing::ElementsAreArray;

class TransposeMatrixTest : public ::testing::Test {
protected:
    int m, n;
    std::vector<std::vector<float>> arr_input;
    std::vector<std::vector<float>> result;

    virtual void SetUp() override {}

    virtual void TearDown() override {}

    void initializeMatrix(int rows, int cols) {
        m = rows;
        n = cols;
        arr_input.assign(m, std::vector<float>(n));
        result.assign(n, std::vector<float>(m));

        int counter = 1;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr_input[i][j] = static_cast<float>(counter++);
    }

    std::vector<float> flattenMatrix(const std::vector<std::vector<float>>& matrix) {
        std::vector<float> flat;
        for (const auto& row : matrix)
            flat.insert(flat.end(), row.begin(), row.end());
        return flat;
    }

    float** convertToPointerArray(std::vector<std::vector<float>>& matrix) {
        float** ptr = new float*[matrix.size()];
        for (size_t i = 0; i < matrix.size(); ++i)
            ptr[i] = matrix[i].data();
        return ptr;
    }

    void freePointerArray(float** ptr) {
        delete[] ptr;
    }
};

TEST_F(TransposeMatrixTest, SquareMatrix_ShouldWork) {
    initializeMatrix(2, 2);
    
    float** inputPtr = convertToPointerArray(arr_input);
    float** resultPtr = convertToPointerArray(result);

    transposeMatrix(m, n, inputPtr, resultPtr);

    std::vector<float> expected = {1, 3, 2, 4};
    EXPECT_THAT(flattenMatrix(result), ElementsAreArray(expected));

    freePointerArray(inputPtr);
    freePointerArray(resultPtr);
}

TEST_F(TransposeMatrixTest, ThinMatrix_ShouldWork) {
    initializeMatrix(4, 2);
    
    float** inputPtr = convertToPointerArray(arr_input);
    float** resultPtr = convertToPointerArray(result);

    transposeMatrix(m, n, inputPtr, resultPtr);

    std::vector<float> expected = {1, 3, 5, 7, 2, 4, 6, 8};
    EXPECT_THAT(flattenMatrix(result), ElementsAreArray(expected));

    freePointerArray(inputPtr);
    freePointerArray(resultPtr);
}

TEST_F(TransposeMatrixTest, FatMatrix_ShouldWork) {
    initializeMatrix(2, 4);
    
    float** inputPtr = convertToPointerArray(arr_input);
    float** resultPtr = convertToPointerArray(result);

    transposeMatrix(m, n, inputPtr, resultPtr);

    std::vector<float> expected = {1, 5, 2, 6, 3, 7, 4, 8};
    EXPECT_THAT(flattenMatrix(result), ElementsAreArray(expected));

    freePointerArray(inputPtr);
    freePointerArray(resultPtr);
}

TEST(HelloTest, BasicAssertions) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}

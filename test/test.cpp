#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <bits/stdc++.h>
using namespace std;
#include "MPI_Syrk_implementation.h"

using ::testing::ElementsAreArray;

TEST(HelloTest, BasicAssertions) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}

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

class GenerateInputTest : public ::testing::Test {
    protected:
        run_config config;
        float** matrix;
    
        virtual void SetUp() override {
            
        }
    
        virtual void TearDown() override {
            for (int i = 0; i < config.m; ++i) {
                delete[] matrix[i];
            }
            delete[] matrix;
        }

        void initializeMatrix(int rows, int cols) {
            config.m = rows;
            config.n = cols;
    
            matrix = new float*[config.m];
            for (int i = 0; i < config.m; ++i) {
                matrix[i] = new float[config.n];
            }
        }
    };

    TEST_F(GenerateInputTest, AllCellsPositive) {

        initializeMatrix(10, 10);

        generate_input(&config, matrix);
    
        for (int i = 0; i < config.m; ++i) {
            for (int j = 0; j < config.n; ++j) {
                EXPECT_GT(matrix[i][j], 0) << "Cell (" << i << ", " << j << ") is not positive.";
            }
        }
    }
    
    TEST_F(GenerateInputTest, AlmostNoCellIsZero) {

        initializeMatrix(10, 10);

        generate_input(&config, matrix);
    
        int zeroCount = 0;
        for (int i = 0; i < config.m; ++i) {
            for (int j = 0; j < config.n; ++j) {
                if (matrix[i][j] == 0) {
                    zeroCount++;
                }
            }
        }
    
        // Check that less than 1% of the cells are zero
        EXPECT_LT(zeroCount, config.m * config.n * 0.01) << "Too many cells are zero.";
    }
    
    TEST_F(GenerateInputTest, HighRandomness) {

        initializeMatrix(10, 10);

        const int numTests = 100;
        std::vector<std::vector<float>> results;
    
        for (int t = 0; t < numTests; ++t) {
            generate_input(&config, matrix);
            std::vector<float> flatMatrix;
            for (int i = 0; i < config.m; ++i) {
                for (int j = 0; j < config.n; ++j) {
                    flatMatrix.push_back(matrix[i][j]);
                }
            }
            results.push_back(flatMatrix);
        }
    
        // Calculate the average value of each cell across all tests
        std::vector<float> averages(config.m * config.n, 0);
        for (const auto& result : results) {
            for (size_t i = 0; i < result.size(); ++i) {
                averages[i] += result[i];
            }
        }
        for (auto& avg : averages) {
            avg /= numTests;
        }
    
        // Check that the standard deviation is high enough to indicate randomness
        std::vector<float> deviations(config.m * config.n, 0);
        for (const auto& result : results) {
            for (size_t i = 0; i < result.size(); ++i) {
                deviations[i] += (result[i] - averages[i]) * (result[i] - averages[i]);
            }
        }
        for (auto& dev : deviations) {
            dev = sqrt(dev / numTests);
        }
    
        // Calculate the average deviation
        float averageDeviation = std::accumulate(deviations.begin(), deviations.end(), 0.0f) / deviations.size();
        // Check that the average deviation is high enough to indicate randomness
        EXPECT_GT(averageDeviation, 2.0) << "Randomness is too low.";
    }
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <bits/stdc++.h>
using namespace std;
#include "MPI_Syrk_implementation.h"

class ParseInputTest : public ::testing::Test {
    protected:
        run_config config;
        int rank = 0;
    
        virtual void SetUp() override {
            // Initialisieren Sie die Konfigurationsstruktur
            config.m = -1;
            config.n = -1;
            config.algo = -1;
            config.result_File = nullptr;
            config.c = -1;
            config.fileName = nullptr;
        }
    
        virtual void TearDown() override {
            // Bereinigen Sie die Konfigurationsstruktur
        }
};

    TEST_F(ParseInputTest, ValidInput) {
        const char* argv[] = {"program", "-a", "1", "-m", "10", "-n", "20", "input.txt"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        parseInput(&config, argc, const_cast<char**>(argv), rank);

        EXPECT_EQ(config.algo, 1);
        EXPECT_EQ(config.m, 10);
        EXPECT_EQ(config.n, 20);
        EXPECT_STREQ(config.fileName, "input.txt");
        SUCCEED();
    }

    TEST_F(ParseInputTest, MissingRequiredParameters) {
        const char* argv[] = {"program", "-a", "1", "-m", "10", "input.txt"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        EXPECT_EXIT(
            parseInput(&config, argc, const_cast<char**>(argv), rank), 
            ::testing::ExitedWithCode(EXIT_FAILURE), 
            "missing parameter n"
        );
    }

    TEST_F(ParseInputTest, InvalidParameter) {
        const char* argv[] = {"program", "-a", "1", "-m", "10", "-n", "20", "-x", "input.txt"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        EXPECT_EXIT(
            parseInput(&config, argc, const_cast<char**>(argv), rank), 
            ::testing::ExitedWithCode(EXIT_FAILURE), 
            "wrong usage: option x doesn't exist"
        );
    }

    TEST_F(ParseInputTest, OptionalParameters) {
        const char* argv[] = {"program", "-a", "1", "-m", "10", "-n", "20", "-o", "result.csv", "-c", "5", "input.txt"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        printf("argc: %d\n", argc);

        parseInput(&config, argc, const_cast<char**>(argv), rank);

        EXPECT_EQ(config.algo, 1);
        EXPECT_EQ(config.m, 10);
        EXPECT_EQ(config.n, 20);
        EXPECT_EQ(config.c, 5);
        EXPECT_STREQ(config.result_File, "result.csv");
        EXPECT_STREQ(config.fileName, "input.txt");
    }

    TEST_F(ParseInputTest, MissingFileName) {
        const char* argv[] = {"program", "-a", "1", "-m", "10", "-n", "20"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        EXPECT_EXIT(
            parseInput(&config, argc, const_cast<char**>(argv), rank), 
            ::testing::ExitedWithCode(EXIT_FAILURE), 
            "missing input file name"
        );
    }
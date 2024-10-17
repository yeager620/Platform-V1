#include "data_loader.h"
#include <random>
#include <algorithm>
#include <iostream>

Database::Database(const std::string& db_name) {
    if (sqlite3_open(db_name.c_str(), &db)) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        exit(EXIT_FAILURE);
    }
}

Database::~Database() {
    sqlite3_close(db);
}

std::vector<std::vector<double>> Database::fetchFeatures(const std::string& table_name) {
    std::vector<std::vector<double>> features;
    std::string sql_query = "SELECT * FROM " + table_name;
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql_query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Failed to fetch data: " << sqlite3_errmsg(db) << std::endl;
        exit(EXIT_FAILURE);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::vector<double> row;
        int num_cols = sqlite3_column_count(stmt) - 1; // Ignore target column
        for (int i = 0; i < num_cols; ++i) {
            row.push_back(sqlite3_column_double(stmt, i));
        }
        features.push_back(row);
    }
    sqlite3_finalize(stmt);
    return features;
}

std::vector<int> Database::fetchTarget(const std::string& table_name) {
    std::vector<int> target;
    std::string sql_query = "SELECT * FROM " + table_name;
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql_query.c_str(), -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Failed to fetch data: " << sqlite3_errmsg(db) << std::endl;
        exit(EXIT_FAILURE);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int target_value = sqlite3_column_int(stmt, sqlite3_column_count(stmt) - 1); // Target column is last
        target.push_back(target_value);
    }
    sqlite3_finalize(stmt);
    return target;
}

void train_test_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_train, std::vector<std::vector<double>>& X_test,
                      std::vector<int>& y_train, std::vector<int>& y_test, double test_size) {
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    int train_size = (1 - test_size) * X.size();

    for (int i = 0; i < X.size(); ++i) {
        if (i < train_size) {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        } else {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }
    }
}

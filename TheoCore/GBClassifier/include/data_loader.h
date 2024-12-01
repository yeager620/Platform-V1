//
// Created by Evan Yeager on 10/17/24.
//

#ifndef MAQUOKETA_PLATFORM_V1_DATA_LOADER_H
#define MAQUOKETA_PLATFORM_V1_DATA_LOADER_H

#include <vector>
#include <string>
#include <sqlite3.h>

class Database {
public:
    sqlite3* db;
    Database(const std::string& db_name);
    ~Database();

    std::vector<std::vector<double>> fetchFeatures(const std::string& table_name);
    std::vector<int> fetchTarget(const std::string& table_name);
};

void train_test_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_train, std::vector<std::vector<double>>& X_test,
                      std::vector<int>& y_train, std::vector<int>& y_test, double test_size=0.2);

#endif //MAQUOKETA_PLATFORM_V1_DATA_LOADER_H

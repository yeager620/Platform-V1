#include <iostream>
#include <vector>
#include "data_loader.h"
#include "gbc.h"

int main() {
    // Connect to the database and load data
    Database db("example.db");
    std::vector<std::vector<double>> X = db.fetchFeatures("my_table");
    std::vector<int> y = db.fetchTarget("my_table");

    // Train-test split
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(X, y, X_train, X_test, y_train, y_test);

    // Train the Gradient Boosting Classifier
    GradientBoostingClassifier gbc(100, 0.1);
    gbc.fit(X_train, y_train);

    // Evaluate the model
    double accuracy = gbc.evaluate(X_test, y_test);
    std::cout << "Model Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}
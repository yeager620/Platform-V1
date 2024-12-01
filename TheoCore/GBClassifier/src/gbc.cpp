#include "gbc.h"
#include <cmath>
#include <limits>

void DecisionStump::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& residuals) {
    int num_features = X[0].size();
    double min_error = std::numeric_limits<double>::max();

    for (int feature = 0; feature < num_features; ++feature) {
        std::vector<double> feature_values;
        for (const auto& row : X) {
            feature_values.push_back(row[feature]);
        }

        std::vector<double> unique_thresholds = feature_values;
        std::sort(unique_thresholds.begin(), unique_thresholds.end());
        unique_thresholds.erase(std::unique(unique_thresholds.begin(), unique_thresholds.end()), unique_thresholds.end());

        for (const double& thresh : unique_thresholds) {
            double left_sum = 0, right_sum = 0;
            int left_count = 0, right_count = 0;

            for (int i = 0; i < X.size(); ++i) {
                if (X[i][feature] <= thresh) {
                    left_sum += residuals[i];
                    left_count++;
                } else {
                    right_sum += residuals[i];
                    right_count++;
                }
            }

            double left_mean = left_count ? left_sum / left_count : 0;
            double right_mean = right_count ? right_sum / right_count : 0;
            double error = left_sum * left_sum + right_sum * right_sum;

            if (error < min_error) {
                min_error = error;
                feature_index = feature;
                threshold = thresh;
                left_value = left_mean;
                right_value = right_mean;
            }
        }
    }
}

double DecisionStump::predict(const std::vector<double>& X) const {
    return X[feature_index] <= threshold ? left_value : right_value;
}

GradientBoostingClassifier::GradientBoostingClassifier(int n_estimators, double learning_rate)
        : n_estimators(n_estimators), learning_rate(learning_rate) {}

void GradientBoostingClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int m = X.size();
    std::vector<double> F(m, 0.0); // Initial predictions are 0

    for (int i = 0; i < n_estimators; ++i) {
        std::vector<double> residuals(m);
        for (int j = 0; j < m; ++j) {
            residuals[j] = y[j] - sigmoid(F[j]);
        }

        DecisionStump stump;
        stump.fit(X, residuals);
        estimators.push_back(stump);

        for (int j = 0; j < m; ++j) {
            F[j] += learning_rate * stump.predict(X[j]);
        }
    }
}

double GradientBoostingClassifier::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<int> GradientBoostingClassifier::predict(const std::vector<std::vector<double>>& X) const {
    int m = X.size();
    std::vector<double> F(m, 0.0);

    for (const auto& stump : estimators) {
        for (int i = 0; i < m; ++i) {
            F[i] += learning_rate * stump.predict(X[i]);
        }
    }

    std::vector<int> predictions(m);
    for (int i = 0; i < m; ++i) {
        predictions[i] = sigmoid(F[i]) >= 0.5 ? 1 : 0;
    }

    return predictions;
}

double GradientBoostingClassifier::evaluate(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    std::vector<int> predictions = predict(X);
    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }
    return (double)correct / y.size();
}

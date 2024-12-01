//
// Created by Evan Yeager on 10/17/24.
//

#ifndef MAQUOKETA_PLATFORM_V1_GBC_H
#define MAQUOKETA_PLATFORM_V1_GBC_H

#include <vector>

class DecisionStump {
public:
    int feature_index;
    double threshold;
    double left_value;
    double right_value;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& residuals);
    double predict(const std::vector<double>& X) const;
};

class GradientBoostingClassifier {
public:
    int n_estimators;
    double learning_rate;
    std::vector<DecisionStump> estimators;

    GradientBoostingClassifier(int n_estimators, double learning_rate);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    double evaluate(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

private:
    double sigmoid(double x) const;
};

#endif //MAQUOKETA_PLATFORM_V1_GBC_H
#ifndef ADABOOST_E_PARALLELL_H
#define ADABOOST_E_PARALLELL_H

#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
#include <algorithm>
#include <math.h>
#include <set>

using namespace std;


struct Decision_Function {
    // Returned by get_feature_threshold_curr_feature with feature_index = -1
    int feature_index = -1;
    double error = INT_MAX;
    double threshold = INT_MIN;
    int direction = 1;
    // vector<double> predictions;

};

struct Decision_Stump {
    // The decision stump of each weak classifier.
    Decision_Function decision_function;
    double alpha_t;

};

Decision_Function min_dec_function(Decision_Function curr_decision_function,Decision_Function in_decision_function )
{
    if (curr_decision_function.error < in_decision_function.error) {
        return curr_decision_function;
    } else {
        return in_decision_function;
    }
}

vector<vector<double> > transpose(vector<vector<double> > &ds) {
    //  Transposes the data set, done so that slicing feature wise is easier
    vector<vector<double> > tr(ds[0].size(), vector<double>(ds.size(), 0));
    for (int i = 0; i < ds.size(); ++i) {
        for (int j = 0; j < ds[i].size(); ++j) {
            tr[j][i] = ds[i][j];
        }
    }
    return tr;

}


class DecisionStump {

public:


    Decision_Function
    get_error_curr_feature_val(vector<int> &labels, vector<double> &weights, vector<double> &feature_vals,
                               double split_val,int feat_index) {

        // Returns the error for the threshold value (split_Val ) as well as decision direction
        double curr_error_1 = 0;
        double curr_error_2 = 0;


        for (int i = 0; i < weights.size(); ++i) {
            if (((feature_vals[i] <= split_val) and (labels[i] != -1)) or
                ((feature_vals[i] > split_val) and (labels[i] != 1))) {
                curr_error_1 += weights[i];
            }
        }

        Decision_Function sol;
        sol.threshold = split_val;
        sol.feature_index=feat_index;
        if (curr_error_1 > 0.5) {
            sol.error = 1 - curr_error_1;
            sol.direction = -1;
        } else {
            sol.error = curr_error_1;
            sol.direction = 1;

        }
        return sol;


    }

    Decision_Function get_feature_threshold_curr_feature(vector<int> &labels, vector<double> &weights,
                                                         vector<double> &feature_vals,
                                                         vector<double> &feature_thresholds,int feat_index) {

        // Returns the best threshold for the current feature index
        Decision_Function best_feature_threshold;

        #pragma omp declare reduction \
        (rwz:Decision_Function:omp_out=min_dec_function(omp_out,omp_in))

        #pragma omp parallel for reduction(rwz:best_feature_threshold)
        for (int i = 0; i < feature_thresholds.size(); ++i) {
            Decision_Function curr_feature_threshold = get_error_curr_feature_val(labels, weights, feature_vals,
                                                                             feature_thresholds[i],feat_index);

            best_feature_threshold = min_dec_function(best_feature_threshold,curr_feature_threshold);
        }
        return best_feature_threshold;
    }



    Decision_Stump fit(vector<int> &labels, vector<vector<double> > &feature_vals,
                       vector<double> &weights, vector<vector<double> > &unique_feature_vals) {
        // Returns the best feature stump for the current set of weights
        Decision_Stump best_feature_stump;
        Decision_Function best_decision_function;


        for (int i = 0; i < feature_vals.size(); ++i) {
            Decision_Function curr_decision_function = get_feature_threshold_curr_feature(labels, weights,
                                                                                          feature_vals[i],
                                                                                          unique_feature_vals[i],i);


            best_decision_function = min_dec_function(best_decision_function,curr_decision_function);

        }

        best_feature_stump.decision_function=best_decision_function;
        return best_feature_stump;

    }


};

class AdaBoost {

    vector<Decision_Stump> weak_classifiers;
public:
    void fit(vector<vector<double> > &X, vector<int> &labels, int t) {
        vector<vector<double> > ds_t = transpose(X);
        // The threshold values as cached in the unique_feature_vals
        vector<vector<double> > unique_feature_vals = get_feature_split_vals(ds_t);

        weak_classifiers = fit_weak_classifiers(labels, ds_t, t, unique_feature_vals);

    }

    vector<int> predict(vector<vector<double> > &X) {
        // Predict function

        vector<int> pred_labels(X.size(), 1);
        // cout << "Started Prediction\n";
        for (int i = 0; i < X.size(); ++i) {
            double negative_label_weight = 0;
            double positive_label_weight = 0;

            for (int j = 0; j < weak_classifiers.size(); ++j) {

                Decision_Stump curr_classifier = weak_classifiers[j];
                double curr_feat_val = X[i][curr_classifier.decision_function.feature_index];
                if (
                        ((curr_feat_val <= curr_classifier.decision_function.threshold) and
                         (curr_classifier.decision_function.direction == 1))\
 or \
                            ((curr_feat_val > curr_classifier.decision_function.threshold) and
                             (curr_classifier.decision_function.direction != 1))
                        ) {
                    negative_label_weight += curr_classifier.alpha_t;
                } else {
                    positive_label_weight += curr_classifier.alpha_t;
                }

            }

            if (negative_label_weight >= positive_label_weight) {
                pred_labels[i] = -1;
            } else {
                pred_labels[i] = 1;
            }
        }
        return pred_labels;
    }

    vector<vector<double> > get_feature_split_vals(vector<vector<double> > &feature_vals) {
        // Returns the unique midpoint threshold 2d vector for each feature.
        vector<vector<double> > solution;
        int i, j, k;
        for (i = 0; i < feature_vals.size(); ++i) {
            vector<double> v(feature_vals[i].begin(), feature_vals[i].end());
            sort(v.begin(), v.end());
            vector<double> midpoint;
            midpoint.push_back(v[0]-1);
            j = 0;
            while (j < v.size() - 1){
                k = j + 1;
                while(v[j] == v[k] and k < v.size() - 1){
                    k += 1;
                }
                midpoint.push_back((v[j] + v[k]) / 2);
                j = k;
            }
            solution.push_back(midpoint);
        }
        return solution;


    }


    vector<Decision_Stump> fit_weak_classifiers(vector<int> &labels, vector<vector<double> > &ds_t, int t, vector<vector<double> > &unique_feature_vals) {
        // fit function called by adaboost
        DecisionStump dec_stump;

        vector<Decision_Stump> weak_classifiers;

        vector<double> weights(labels.size(), 1.0 / labels.size());

        for (int i = 0; i < t; ++i) {
            Decision_Stump curr_decision_stump = dec_stump.fit(labels, ds_t, weights, unique_feature_vals);

            curr_decision_stump.alpha_t = 0.5 * log((1 - curr_decision_stump.decision_function.error)/(curr_decision_stump.decision_function.error));

            if (curr_decision_stump.decision_function.error <= 0.00005)
                return weak_classifiers;


            update_weights(curr_decision_stump, labels, weights, ds_t[curr_decision_stump.decision_function.feature_index]);
            weak_classifiers.push_back(curr_decision_stump);

        }

        return weak_classifiers;

    }

    double update_weights(Decision_Stump ds, vector<int> &labels, vector<double> &weights, vector<double> &feature_vals) {
        // Updates the weight
        double alpha_t = ds.alpha_t;
        double weights_sum = 0;

        for (int i = 0; i < weights.size(); ++i) {
            if (((feature_vals[i] <= ds.decision_function.threshold) and (ds.decision_function.direction == 1)) or
                ((feature_vals[i] > ds.decision_function.threshold) and (ds.decision_function.direction == -1))) {
                weights[i] = weights[i] * exp(alpha_t * labels[i]);
            } else {
                weights[i] = weights[i] * exp(-1 * alpha_t * labels[i]);
            }
            weights_sum += weights[i];
        }
        for (int i = 0; i < weights.size(); ++i) {
            weights[i] = weights[i] / weights_sum;
        }
        return alpha_t;
    }

};


#endif



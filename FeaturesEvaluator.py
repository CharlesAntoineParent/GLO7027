from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
import shap
import seaborn as sns
import pandas as pd

class FeaturesEvaluator:
    """[summary] https://mljar.com/blog/feature-importance-xgboost/
    """
    def __init__(self, p_fitted_model, p_X_train, p_Y_train, p_X_test, p_Y_test):
        self.fitted_model = p_fitted_model
        self.X_train = p_X_train
        self.X_test = p_X_test
        self.Y_train = p_Y_train
        self.Y_test = p_Y_test
        self.SHAP_explainer = shap.TreeExplainer(self.fitted_model)
        self.SHAP_values = self.SHAP_explainer.shap_values(self.X_test)
        self.perm_importance = permutation_importance(self.fitted_model, self.X_test, self.Y_test)

    
    def get_model(self):
        return  self.fitted_model
    
    def get_X_train(self):
        return self.X_train 
    
    def get_X_test(self):
        return self.X_test 

    def get_Y_train(self):
        return self.Y_train

    def get_Y_test(self):
        return self.Y_test
    
    def get_r2_score(self):
        return self.fitted_model.score(self.X_test, self.Y_test)

    def get_perm_importance(self):
        return self.perm_importance

    def get_sorted_importance(self, plot = True):
        sorted_idx = self.fitted_model.feature_importances_.argsort()

        if plot:
            plt.barh(self.X_test.keys()[sorted_idx], self.fitted_model.feature_importances_[sorted_idx])
            plt.xlabel("Xgboost Feature Importance")
            plt.show()

        return {'feature':self.X_test.keys()[sorted_idx], 'importance': self.fitted_model.feature_importances_[sorted_idx]}

    def get_sorted_permutaion_importance(self, plot = True):
        sorted_idx = self.perm_importance.importances_mean.argsort()

        if plot:
            plt.barh(self.X_test.keys()[sorted_idx], self.perm_importance.importances_mean[sorted_idx])
            plt.xlabel("Permutation Importance")
            plt.show()

        return {'feature':self.X_test.keys()[sorted_idx], 'importance_permutation': self.perm_importance.importances_mean[sorted_idx]}

    def get_correlation_heatmap(self):
        sorted_idx = self.perm_importance.importances_mean.argsort()
        train = self.X_train[self.X_train.keys()[sorted_idx]]
        correlations = train.corr()

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
        plt.show();

    


    def get_sorted_mean_SHAP_values(self):
        mean_array_SHAP = self.SHAP_values.mean(axis = 0)
        sorted_idx =mean_array_SHAP.argsort()
        return {'feature':self.X_test.keys()[sorted_idx], 'SHAP_value': mean_array_SHAP[sorted_idx]}

    def get_df_SHAP_values(self):
        df = pd.DataFrame(self.SHAP_values, columns=self.X_test.keys(), index=self.X_test.keys())
        return df


    def get_SHAP_values_bar_plot(self):
        shap.summary_plot(self.get_SHAP_values(), self.X_test, plot_type="bar")

    def get_SHAP_values_impact_model_output(self):
        shap.summary_plot(self.get_SHAP_values(), self.X_test, plot_type="bar")

    def get_dependence_plot(self, p_feature):
        shap.dependence_plot(p_feature, self.SHAP_values, self.X_test)
# import modules
import os
import pickle
import xgboost as xgb
import lightgbm as lgb
from config import GetDict


# store feature information:
numeric_features = GetDict('feature_info.json')['numeric_features']
categorical_features = GetDict('feature_info.json')['categorical_features']
all_columns = numeric_features + categorical_features
dropped_columns = GetDict('feature_info.json')['dropped_columns']

def FeatureImportance(model):
        
        """Performs Feature Importance for modeling and return mean squared error

        Parameters
        ----------
        model : string
            model name 
        
        Return
        ----------
        plot
            feature importance plot

        """
        
        if model == 'RandomForest':
            load_model = pickle.load(
                open(os.path.join('../models', f'{model}.sav'), 'rb'))
            
            feature_names = list(numeric_features) + list(load_model.named_steps.columntransformer.transformers_[
                1][1][1].get_feature_names(categorical_features))
            
            feat_importances = pd.Series(load_model.steps[1][1].feature_importances_, index=feature_names)
            
            plot = feat_importances.nlargest(15).plot(kind='barh', title=f"{model} Feature Importance")

        elif model == 'XGBoost':
            load_model = pickle.load(
                open(os.path.join('../models', f'{model}.sav'), 'rb'))

            feature_names = list(numeric_features) + list(load_model.named_steps.columntransformer.transformers_[
                1][1][1].get_feature_names(categorical_features))

            plot = xgb.plot_importance(
                load_model.steps[1][1], max_num_features=15, title=f"{model} Feature Importance").set_yticklabels(feature_names)

        elif model == 'LGBoost':
            load_model = pickle.load(
                open(os.path.join('../models', f'{model}.sav'), 'rb'))

            feature_names = list(numeric_features) + list(load_model.named_steps.columntransformer.transformers_[
                1][1][1].get_feature_names(categorical_features))
            
            plot = lgb.plot_importance(
                load_model.steps[1][1], max_num_features=15, title=f"{model} Feature Importance").set_yticklabels(feature_names)
        else:
            print(f'{model} not found')
            
        return plot
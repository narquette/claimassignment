import pandas as pd
import warnings
import numpy as np
import logging
import datetime
import time 
import pickle
import os
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split,  cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import TransformedTargetRegressor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
from config import GetDict

# set dictionary for data type 
data_types = {"Claim.Number": np.float,
 "Claim.Line.Number": np.object,
 "Member.ID": np.object,
 "Provider.ID": np.object,
 "Line.Of.Business.ID": np.object,
 "Revenue.Code": np.object,
 "Service.Code": np.object,
 "Place.Of.Service.Code": np.object,
 "Procedure.Code": np.object,
 "Diagnosis.Code": np.object,
 "Claim.Charge.Amount": np.float,
 "Denial.Reason.Code": np.object,
 "Price.Index": np.object,
 "In.Out.Of.Network": np.object,
 "Reference.Index": np.object,
 "Pricing.Index": np.object,
 "Capitation.Index": np.object,
 "Subscriber.Payment.Amount": np.float,
 "Provider.Payment.Amount": np.float,
 "Group.Index": np.object,
 "Subscriber.Index": np.object,
 "Subgroup.Index": np.object,
 "Claim.Type": np.object,
 "Claim.Subscriber.Type": np.object,
 "Claim.Pre.Prince.Index": np.object,
 "Claim.Current.Status": np.object,
 "Network.ID": np.object,
 "Agreement.ID": np.object}

# set dictionaries for tuning models
random_tune = GetDict('random_tune.json')
xgb_tune = GetDict('xgb_tune.json')
lgb_tune = GetDict('lgb_tune.json')

# #store feature information:
numeric_features = GetDict('feature_info.json')['numeric_features']
categorical_features = GetDict('feature_info.json')['categorical_features']
all_columns = numeric_features + categorical_features
dropped_columns = GetDict('feature_info.json')['dropped_columns']

class MachineLearning():
    
    """
    
    A class used to perform machines learning on regression models for Linear, RandomForest, Xgboost, and Lgboost regression modeling 
    using the sklearn pipe processing including logging for each model

    """
    
    def __init__(self, train_data=None, test_data=None, label=None, log_file=None, imbalance=None):
        
        """
        Parameters
        ----------
        train_data : str
            the file path representation of where the data resides
        test_data : str
            the file path representation of where the data resides
        label : str
            the name of the label, target, or dependent variable
        log_file : str
            the name of the log file
            
        Attributes
        ----------
        train_data : object
            put the data into a dataframe
        X : str
            the features for the dataset    
        Y :
            the label data for the dataset
        start : time
            start time for logging purposes
        now : datetime
            insert date time into filename
        file_name : str
            build file name for logging (one per model)
        logging : obj
            start logging for the current model
    
        """
        
        # load and split TRAIN DATA
        if train_data and label:
            
            # put data into memory and establish label for TRAIN DATA
            self.train = train_data
            self.label = label
            self.train_data = pd.read_csv(self.train, low_memory=False)
            
            # set X and y for TRAIN
            self.X = self.train_data.drop(self.label, axis=1)
            self.y = self.train_data[[self.label]]
            
            #if imbalance: TODO remove
            #    under = RandomUnderSampler(random_state=42)
            #    self.X, self.y = under.fit_resample(self.X, self.y)

            # Split Train and Test for TRAIN
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                        self.y,
                                                        test_size=0.33,
                                                        random_state=42,
                                                        stratify=self.y)
            
        # load test data for prediction
        if test_data:
            self.test = test_data
            self.test_data = pd.read_csv(self.test, low_memory=False)
       
        # Set log file name and start time
        self.start = time.time()
        now = datetime.datetime.now()
        self.file_name = f"{log_file}_{now.year}_{now.month}_{now.day}.log"
        
        # Start Logging
        self.logging = logging.basicConfig(format='%(asctime)s %(message)s', filename=self.file_name, level=logging.DEBUG)
        
        
    def PreProcessing(self):
        
        """Establishes preprocessing and a pipeline for modeling

        Parameters
        ----------
        None 

        """
        
        # set transformer for numeric features
        numeric_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())])

        # set transformer for categorical features
        categorical_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        # Make Transformer
        self.preprocessing = ColumnTransformer(
                                    transformers=[
                                        ('num', numeric_transformer, numeric_features),
                                        ('cat', categorical_transformer, categorical_features)])
        
        return self.preprocessing
       
        
    
    def CrossValidation(self):
        
        """Performs cross validations for modeling and return mean squared error

        Parameters
        ----------
        None
        
        Return
        ----------
        mse (Mean Squared Error )

        """
        
        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())
        #rus.fit_resample(X, y)        
            
        # start logging
        start = time.time()
        logging.info(f"{self.model_name} Start")
        
        # model
        def mse(y_true, y_pred): return mean_squared_error(y_true, y_pred)
          
        # evaluate model
        # cross_score = cross_val_score(self.pipe, self.X, self.y, scoring='neg_mean_absolute_error', cv=None, n_jobs=-1)
        scoring = { 'mse': make_scorer(mse),
                    'accuracy' : make_scorer(accuracy_score),
                    'precision' : make_scorer(precision_score),
                    'recall' : make_scorer(recall_score),
                    'f1' : make_scorer(f1_score) }
        
        mse = cross_validate(self.pipe, self.X, self.y.values.ravel(), cv=5, scoring=scoring)
               
        # set log for finishing
        logging.info(f"Score for {self.model_name} is {mse}")
        logging.info(f"Run Time for {self.model_name} is {(time.time() - self.start) // 60} minutes")
        
        # close logging file
        logging.FileHandler(self.file_name).close()
        
        return (mse['test_mse'], mse['test_accuracy'], mse['test_precision'], mse['test_recall'], mse['test_f1'])
        
    def Prediction(self, model_name):
        
        """Establishes a scoring (mean squared error) for modeling

        Parameters
        ----------
        model_name : string
            The model name will be passed into the function and used to provide a prediction and write the information to file

        Return
        ----------
        None
        
        """
        
        # load model
        load_model = pickle.load(open(os.path.join('../models', model_name), 'rb'))
        
        # store results in a dataframe
        results_df = pd.DataFrame()
        results_df['Claim Number'] = self.test_data['Claim.Number']
        results_df['Claim Line Number'] = self.test_data['Claim.Line.Number']
        results_df['Observed UnPaid Status'] = self.test_data['UnpaidClaim']
       
        # drop columns prior to prediction
        # needs to happen after load results dataframe as columns mentioned above (e.g. Claim number) are dropped
        self.test_data.drop(dropped_columns, inplace=True, axis=1)
        
        # perform predictions                         
        predictions = load_model.predict(self.test_data)
        
        # add predictions to results dataframe
        results_df['Predicted UnPaid Status'] = predictions
            
        # add results to a file
        results_df.to_csv(os.path.join('../data','prediction','test_unpaid_procedures.tar.gz'), compression='gzip', index=False)
        
        # set log for finishing
        logging.info(f"Prediction Time for {model_name} is {time.time() - self.start} seconds")
                                 
    
    def Scoring(self, save_model=False):
        
        """Establishes a scoring (mean squared error) for modeling

        Parameters
        ----------
        save_model : boolean, optional
            If true, the model will be save and prediction function will be called

        Return
        ----------
        mse (Mean Squared Error )
        
        """
        
        # start logging
        start = time.time()
        logging.info(f"{self.model_name} Start")
                                 
        # set mse to None
        # need to allow for saving model and predicting
        mse = None

        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())
        
        # save model if save_mode is True:        
        if save_model:
            mse = None
            score = None
            report = None
            pickle.dump(self.pipe, open(os.path.join('../models',f"{self.model_name}.sav"), 'wb'))    
    
        else :
                                 
            # Mean Square Error Info
            predictions = self.pipe.predict(self.X_test)
            actual = self.y_test
            mse = mean_squared_error(actual, predictions)
            report = classification_report(actual, predictions, output_dict=True)
            score = self.pipe.score(self.X_test, self.y_test.values.ravel())

        # set log for finishing
        logging.info(f"Score for {self.model_name} is {mse}")
        logging.info(f"Run Time for Linear Regression is {(time.time() - self.start) // 60} minutes")
        
        # close logging file
        logging.FileHandler(self.file_name).close()
        
        return (mse, score, report)
       

    def LinearRegression(self, cross_validation=False, prediction=False, save_model=False):
        
        """Performs linear regression

        Parameters
        ----------
        cross_validation : boolean, optional
            if True it will perform cross validation, otherwise it will perform normal scoring
        prediction : boolean, optional
            if true, the scoring function will be called that saves the model and performs a prediction on all of the test data            
            
        Return
        ----------
        mse (Mean Squared Error )
        
        """

        # set model name
        self.model_name = 'Linear Regression'
               
        # put model into an object
        linear_regression = LinearRegression()

        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, linear_regression)        

        # performs cross validation
        if cross_validation:
            
            mse = self.CrossValidation()

        # perform a prediction
        elif prediction:
            
            mse = self.Scoring(save_model=save_model)
        
        # normal scoring for tuning
        else:
            
            mse = self.Scoring(save_model=save_model)
            
        # return best score
        return mse
        
    def LogisticRegression(self, cross_validation=False, prediction=False, save_model=False):
        
        """Performs linear regression

        Parameters
        ----------
        cross_validation : boolean, optional
            if True it will perform cross validation, otherwise it will perform normal scoring
        prediction : boolean, optional
            if true, the scoring function will be called that saves the model and performs a prediction on all of the test data            
            
        Return
        ----------
        mse (Mean Squared Error )
        
        """

        # set model name
        self.model_name = 'Logistic Regression'
               
        # put model into an object
        logistic_regression = LogisticRegression(max_iter=500)

        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, logistic_regression)        

        # performs cross validation
        if cross_validation:
            
            mse = self.CrossValidation()

        # perform a prediction
        elif prediction:
            
            mse = self.Scoring(save_model=save_model)
        
        # normal scoring for tuning
        else:
            
            mse = self.Scoring(save_model=save_model)
            
        # return best score
        return mse
        
    def RandomForest(self, parameter_dict, regressor=True, cross_validation=False, prediction=False, save_model=False):
        
        """Perform random forest modeling and returns mean squared error
        
        Parameters
        ----------
        parameter_dict : dictionary
            passes in parameters for tuning or cross validations
        cross_validation : boolean, optional
            if True it will perform cross validation, otherwise it will perform normal scoring
        prediction : boolean, optional
            if true, the scoring function will be called that saves the model and performs a prediction on all of the test data

        Return
        ----------
        mse (Mean Squared Error )
        
        """
        
        # set model name
        self.model_name = 'RandomForest'
        
        # put model into an object
        if regressor:        
            random_forest = RandomForestRegressor(criterion='mse',
                                                   oob_score=True, 
                                                   n_jobs=-1, 
                                                   random_state=44,
                                                   bootstrap=True,
                                                   max_samples=0.2)
        else:
            random_forest = RandomForestClassifier(criterion='gini',
                                                   oob_score=True, 
                                                   n_jobs=-1, 
                                                   random_state=44,
                                                   bootstrap=True,
                                                   max_samples=0.2)
        
        # set random forest parameter
        random_forest.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, random_forest)        

        # performs cross validation
        if cross_validation:
            
            mse = self.CrossValidation()

        # perform a prediction
        elif prediction:
            
            mse = self.Scoring(save_model=save_model)
        
        # normal scoring for tuning
        else:
            
            mse = self.Scoring(save_model=save_model)
            
        # return best score
        return mse
        

    def XGboost(self, parameter_dict, regressor=True, cross_validation=False, prediction=False, save_model=False):
        
        """Perform XGboost forest modeling and returns mean squared error
        
        Parameters
        ----------
        parameter_dict : dictionary
            passes in parameters for tuning or cross validations
        cross_validation : boolean, optional
            if True it will perform cross validation, otherwise it will perform normal scoring
        prediction : boolean, optional
            if true, the scoring function will be called that saves the model and performs a prediction on all of the test data            

        Return
        ----------
        mse (Mean Squared Error )
        
        """
        
        # set model name
        self.model_name = 'XGBoost'
        
        if regressor:
            # put model into an object
            xg_boost = xgb.XGBRegressor(base_score=0.5, 
                                    booster='gbtree', 
                                    colsample_bylevel=1,
                                    colsample_bytree=1, 
                                    max_delta_step=0,
                                    missing=None, 
                                    n_jobs=-1,
                                    nthread=None, 
                                    objective='reg:squarederror', 
                                    random_state=44,
                                    reg_alpha=0, 
                                    reg_lambda=1, 
                                    scale_pos_weight=1, 
                                    seed=None,
                                    silent=True, 
                                    subsample=1)
            
        else:
            # put model into an object
            xg_boost = xgb.XGBClassifier(base_score=0.5, 
                                    booster='gbtree', 
                                    colsample_bylevel=1,
                                    colsample_bytree=1, 
                                    max_delta_step=0,
                                    missing=None, 
                                    n_jobs=-1,
                                    nthread=None, 
                                    objective='binary:logistic', 
                                    random_state=44,
                                    reg_alpha=0, 
                                    reg_lambda=1, 
                                    scale_pos_weight=1, 
                                    seed=None,
                                    silent=True, 
                                    subsample=1)
        
        # set random forest parameter
        xg_boost.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, xg_boost)        

        # performs cross validation
        if cross_validation:
            
            mse = self.CrossValidation()

        # perform a prediction
        elif prediction:
            
            mse = self.Scoring(save_model=save_model)
        
        # normal scoring for tuning
        else:
            
            mse = self.Scoring(save_model=save_model)
            
        # return best score
        return mse

    def LGboost(self, parameter_dict, regressor=True, cross_validation=False, prediction=False, save_model=False):
        
        """Perform LGBoost forest modeling and returns mean squared error
        
        Parameters
        ----------
        parameter_dict : dictionary
            passes in parameters for tuning or cross validations
        cross_validation : boolean, optional
            if True it will perform cross validation, otherwise it will perform normal scoring
        prediction : boolean, optional
            if true, the scoring function will be called that saves the model and performs a prediction on all of the test data

        Return
        ----------
        mse (Mean Squared Error )
        
        """
        
        # set model name
        self.model_name = 'LGBoost'
        
        if regressor:
            # put model into an object
            lg_boost = lgb.LGBMRegressor(boosting_type='gbdt', 
                                    feature_fraction=.8, 
                                    n_jobs=-1,
                                    nthread=None, 
                                    objective='regression', 
                                    random_state=44,
                                    scale_pos_weight=1, 
                                    bagging_fraction=1)
        else:
            # put model into an object
            lg_boost = lgb.LGBMClassifier(boosting_type='gbdt', 
                                    feature_fraction=.8, 
                                    n_jobs=-1,
                                    nthread=None, 
                                    objective='binary', 
                                    random_state=44,
                                    scale_pos_weight=1, 
                                    bagging_fraction=1)
            
        
        # set random forest parameter
        lg_boost.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, lg_boost)     
        
        # performs cross validation
        if cross_validation:
            
            mse = self.CrossValidation()

        # perform a prediction
        elif prediction:
            
            mse = self.Scoring(save_model=save_model)
        
        # normal scoring for tuning
        else:
            
            mse = self.Scoring(save_model=save_model)
            
        # return best score
        return mse
    



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

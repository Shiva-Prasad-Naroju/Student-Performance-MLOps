# import os
# import sys

# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path =  os.path.join('artifacts','preprocessor.pkl')

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_obj(self):

#         '''
#         This function is responsible for data transformation
#         '''
#         try:
#             numerical_columns = ['writing_score','reading_score']
#             categorical_columns = [
#                 'gender',
#                 'race_ethnicity',
#                 'parental_level_of_education',
#                 'lunch',
#                 'test_preparation_course'
#             ]

#             num_pipeline = Pipeline(
#                 steps=[
#                     ('imputer',SimpleImputer(strategy='median')),
#                     ('scaler',StandardScaler(with_mean=False))
#                 ]
#             )

#             cat_pipeline = Pipeline(
#                 steps=[
#                     ('imputer',SimpleImputer(strategy='most_frequent')),
#                     ('one_hot_encoder',OneHotEncoder()),
#                     ('scaler',StandardScaler(with_mean=False))
#                 ]
#             )

#             logging.info(f'Numerical columns : {numerical_columns}')
#             logging.info(f'Categorical columns : {categorical_columns}')

#             preprocessor = ColumnTransformer(
#                 [
#                     ('num_pipeline',num_pipeline,numerical_columns),
#                     ('cat_pipelines',cat_pipeline,categorical_columns)
#                 ]
#             )
#             return preprocessor
    
#         except Exception as e:
#             raise CustomException(e,sys)
    
#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info('Read train and test data completed')
#             logging.info('Obtaining preprocessing object')

#             preprocessing_obj = self.get_data_transformer_obj()

#             target_column_name = 'math_score'
            
#             numerical_columns = ['writing_score','reading_score']

#             input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')

#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]

#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
#             logging.info('Saved preprocessing object')

#             save_object(
#                 file_path = self.data_transformation_config.preprocessor_obj_file_path,
#                 obj = preprocessing_obj
#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path

#             )

#         except Exception as e:
#             raise CustomException(e,sys)

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
























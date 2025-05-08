import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # use to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            '''
            This block is mainly responsible for data transformation
            
            
            '''
            numerical_feature = ['writing_score','reading_score']
            categorical_feature = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical column encoding complete")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical column encoding complete")

            logging.info(f"cat columns:{categorical_feature}")
            logging.info(f"num columns:{numerical_feature}")
            #ColumnTransformer is used to combine numerical and categorical pipeline together
            preprocessor = ColumnTransformer(
                [
                    ("numerical_piipeline",num_pipeline,numerical_feature),
                    ("cat_pipeline",cat_pipeline,categorical_feature)
                ]
            )
            return preprocessor
            # combine numerical and categorical pipeline together usig columntransformer
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data complete")
            logging.info("obtain preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ['writing_score','reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing on training and test dataframe .")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saved preprrocessing object")
            print("saved preprrocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessor_obj)

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
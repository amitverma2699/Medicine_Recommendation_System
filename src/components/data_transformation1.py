import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Customexception
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocessor.pkl')
    label_encoder_obj_file_path = os.path.join('Artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info("Starting data transformation pipeline.")

            # Defining categorical columns (numerical not used in this case)
            categorical_cols = ["Gender", "Symptoms", "Causes", "Disease"]

            # Creating pipelines
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler())
            ])

            # Column transformer to combine pipelines
            preprocessor = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_cols),
            ])

            logging.info("Data transformation pipeline created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error("Exception occurred in get_data_transformation")
            raise Customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation process.")

            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n{test_df.head().to_string()}")

            # Get transformation object
            preprocessing_obj = self.get_data_transformation()

            # Splitting features and target
            target_column_name = 'Medicine'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Encode target column
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            # Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets.")

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save label encoder object
            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=label_encoder
            )

            logging.info("Preprocessing and label encoder pickle files saved successfully.")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise Customexception(e, sys)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation1 import DataTransformation
from src.logger import logging
from src.exception import Customexception
import os
import sys
import pandas as pd

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

preprocessor_obj=data_transformation.initiate_data_transformation(train_data_path,test_data_path)


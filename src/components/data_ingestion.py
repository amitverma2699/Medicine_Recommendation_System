import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import os
import sys

@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join("Artifacts","raw.csv")
    train_data_path : str = os.path.join("Artifacts","train.csv")
    test_data_path : str = os.path.join("Artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            data=pd.read_csv(Path(os.path.join("notebooks\data","Medicine__data.csv")))
            logging.info("Read the Dataset.")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("I have saved the raw data in the artifact folder")

            
            logging.info("Perform Train Test Split")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Train Test Split Completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured in Data Ingestion")
            raise Customexception(e,sys)
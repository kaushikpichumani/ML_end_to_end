import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# there should be input
# where to save train, test and raw data. we will crreate this in a class
# in data ingestion any input will be given through this 

@dataclass
# inside the class to define the class variablewe use init, if we use this decorator we will be able to 
# define your class variable 
class DataIngestionConfig:
    # this will tell where to store the train, test and raw data
    train_data_path :str=os.path.join("artifact","train.csv")
    test_data_path :str=os.path.join("artifact","test.csv")
    raw_data_path :str=os.path.join("artifact","data.csv")

class DataIngestion:
    # if we are just defining variables then go with dataclass else go with __init__
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            df = pd.read_csv("/notebook/data/stud.csv") # we can change here to read from  Mongo db or sql etc
            logging.info('read the dataset as DF')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion is complete")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys) 
            

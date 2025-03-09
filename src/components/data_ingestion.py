import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    # Define file paths for raw, train, and test data
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Reads dataset, splits into train & test sets, and saves them."""
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # This line ensures that the directory where the train dataset will be stored exists 
            # before we try to save the file. we could use the any other paths in artifacts instead of 
            # using train_data_path, that results the same.

            # The script needs to save the train.csv file inside 'artifacts/'.

            # So, this line ensures the 'artifacts/' directory is created if it doesn’t exist,
            # preventing errors.

            # Since all files (train.csv, test.csv, data.csv) are stored in 'artifacts/', 
            # using any one of them would ensure that the directory is created.


            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # Save raw data

            logging.info("Train test split initiated")

            # Savind the train and test datasets in train_set and test_set
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # saving train_set as csv file in the train_data_path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # saving train_set as csv file in the test_data_path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    modeltainer = ModelTrainer()
    print(modeltainer.initiate_model_trainer(train_arr, test_arr))


# Working process:
# We manually provide the dataset path inside the function initiate_data_ingestion()

# initiate_data_ingestion() this function will create splits the data and three files 
# raw_data, train.csv and test.csv in the artifacts folder

# The function will returns the train.csv and test.csv

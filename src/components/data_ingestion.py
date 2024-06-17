import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize DataIngestion Class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Starts")
        
        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'eda_results.csv'))
            logging.info("Dataset read as pandas DataFrame")
            
            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Write the raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("My raw data is being created")
            
            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            
            # Write the train and test data to CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)

# To run the data ingestion process
if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

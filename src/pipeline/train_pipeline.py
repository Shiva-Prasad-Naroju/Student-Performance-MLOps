# Since we orchestrated the entrire workflow in the data_ingestion.py module itself, its redundant to
# use this module or else we can also make of this module.



# import sys
# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# from src.exception import CustomException
# from src.logger import logging

# def train_pipeline():
#     try:
#         logging.info("Starting the training pipeline...")

#         # **Step 1: Data Ingestion**
#         data_ingestion = DataIngestion()
#         train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

#         # **Step 2: Data Transformation**
#         data_transformation = DataTransformation()
#         train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

#         # **Step 3: Model Training**
#         model_trainer = ModelTrainer()
#         r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

#         logging.info(f"Training pipeline completed successfully. Model R² score: {r2_score}")
#         print(f"Training pipeline completed successfully. Model R² score: {r2_score}")

#     except Exception as e:
#         logging.error(f"Error in training pipeline: {str(e)}")
#         raise CustomException(e, sys)

# if __name__ == "__main__":
#     train_pipeline()

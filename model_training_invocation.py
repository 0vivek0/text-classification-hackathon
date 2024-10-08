from prediction import *
from model_training import *
from datetime import datetime
import logging

from datasets import load_dataset

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
model_checkpoint = "distilroberta-base"


def start_training(saved_model_name):
    """
        Prepares data and initiates model training.

        Args:
            saved_model_name: Path to save the trained model.

        This function generates training, validation, and test datasets, then trains the model.
    """

    generate_train_test_val_data("Enhanced_Synthetic_Data_1000.xlsx")

    dataset = load_dataset('csv', data_files={
                                        'train': 'Enhanced_Synthetic_Data_Train.csv',
                                        'valid': 'Enhanced_Synthetic_Data_Val.csv',
                                        'test':  'Enhanced_Synthetic_Data_Test.csv'
                                        },)

    saved_model_name = model_training_and_saving(dataset, saved_model_name,2)

def get_scope_for_query(query, saved_model_name):
    """
        Retrieves the predicted scope for a given query.

        Args:
            query: The input query string.
            saved_model_name: Path to the saved model.

        Returns:
            dict: The predicted scope for the query.
    """
    return dict_in_out[str(get_prediction_for_query(query, saved_model_name))]

def get_prediction_bulk_data():
    """
        Loads dataset and evaluates model accuracy on test data.

        This function loads training, validation, and test datasets and calculates model accuracy.
    """
    dataset = load_dataset('csv', data_files={
                                        'train': 'Enhanced_Synthetic_Data_Train.csv',
                                        'valid': 'Enhanced_Synthetic_Data_Val.csv',
                                        'test':  'Enhanced_Synthetic_Data_Test.csv'
                                        },)
    accuracy = get_test_data_result(dataset, saved_model_name)



if __name__=="__main__":
    # saved_model_name = "fine_tuned_model_bayer_bert_epoch_large_5/"
    saved_model_name = "fine_tuned_model_bayer_bert_1000"
    start_training(saved_model_name)
    query = "get me the python code for data extraction with PyPDF"
    logging.info(f"Scope for query {get_scope_for_query(query, saved_model_name)}")
    print(get_scope_for_query(query, saved_model_name))

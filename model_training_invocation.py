
from text_preprocessing import *
from prediction import *
from model_training import *
from datetime import datetime
import logging

from datasets import load_dataset

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
model_checkpoint = "distilroberta-base"


def start_training(saved_model_name):
    generate_train_test_val_data("Enhanced_Synthetic_Data_1000.xlsx")

    dataset = load_dataset('csv', data_files={
                                        'train': 'Enhanced_Synthetic_Data_Train.csv',
                                        'valid': 'Enhanced_Synthetic_Data_Val.csv',
                                        'test':  'Enhanced_Synthetic_Data_Test.csv'
                                        },)

    saved_model_name = model_training_and_saving(dataset, saved_model_name, 2)

def get_scope_for_query(query, saved_model_name):
    return dict_in_out[str(get_prediction_for_query(query, saved_model_name))]

def get_prediction_bulk_data():
    dataset = load_dataset('csv', data_files={
                                        'train': 'Enhanced_Synthetic_Data_Train.csv',
                                        'valid': 'Enhanced_Synthetic_Data_Val.csv',
                                        'test':  'Enhanced_Synthetic_Data_Test.csv'
                                        },)
    accuracy = get_test_data_result(dataset, saved_model_name)



if __name__=="__main__":
    saved_model_name = "fine_tuned_model_bayer_bert_epoch_large_5/"
    # start_training(saved_model_name)
    query = "get me the python code for data extraction with PyPDF"
    print(get_scope_for_query(query, saved_model_name))

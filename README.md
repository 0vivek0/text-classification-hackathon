#Prompt Injection Detector:-

##Description: Prevent abuse of chatbot by classification of prompts received from user.

##Objective: Aim is to expose a Rest API which takes a prompt and pre-trained model classify text as ‘Allowed’ or ‘Restricted;

###Scenario: Prompt Text classification.

###Technical Implementation:
• Backend: Rest API to be created using FastAPI.
• Detector must not be a LLM
• Train custom model based on BERT / TF-IDF, Word2Vec or model of choice.
• Use the Custom Model to do the classification of the prompt text.
• Training Dataset – will be provided as input along with pre-processor python.
• Requirement is to create a dataSet, train the model, save custom model and use said model to classify queries.

###Tools and Technologies:-
FastAPI
BERT, Huggingface libraries
Python 3
Dependencies: Scikit-learn, FastAPI, Uvicorn, Pandas, matplotlib, logging, sentence transformers etc.


###IDE:-
    • VS code
    • Pycharm

###Components:
    • Pre-processing (Cleaning, Vectorisation, and Tokenisation)
    • Model training and persistance
    • Prediction (Bulk result generation and single query result)
    • Get_scope_api
        ◦ query (string)
        ◦ model name (string)
        ◦ Returns if query is allowed or restricted

Use-case: Prompt Injection Detector:-

Description: Prevent abuse of chatbot by classification prompts received from user as ‘Allowed’ or
‘‘Restricted’.

Objective: Aim is to expose a Rest API which takes a prompt and pre-trained model classify text as
‘Allowed’ or ‘Restricted;
Scenario: Prompt Text classification.
Technical Implementation:
• Backend: Rest API to be created using FastAPI.
• No use LLM
• Train custom model based on BERT / TF-IDF, Word2Vec or model of choice.
• Use the Custom Model to do the classification of the prompt text.
• Training Dataset – will be provided as input along with pre-processor python.
• Requirement is to create DataSet, Train the model, Save Custom Model and use the Custom Model to do Text classification.

Tools and Technologies:-
Backend: FastAPI
Machine Learning: BERT, Huggingface libraries
Language: Python
Dependencies: Scikit-learn, FastAPI, Uvicorn, Pandas, matplotlib, logging, sentence transformers etc.


IDE:-
    • VS code
    • Pycharm

Components:
    • Pre-processing (Cleaning, Vectorisation and Tokenisation)
    • Model Training and storage
    • Prediction (Bulk  result generation and single query result)
    • Get_scope_api
        ◦ query
        ◦ model name
        ◦ Returns query is allowed or not

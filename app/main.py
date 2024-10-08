from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from model_training_invocation import get_scope_for_query

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check")
async def check_api_health():
    """Method to check the service."""
    return {"status": "UP"}


@app.post("/get_scope_for_given_query")
async def get_scope_for_given_query(saved_model_name: str = Form(...), query: str = Form(...)):
    try:
        """
            API endpoint to get the scope for a given query and model.

            Args:
                saved_model_name (str): The name of the saved model.
                query (str): The query string.

            Returns:
                str: The scope of the given query.

            Raises:
                HTTPException: If any error occurs during processing.
            """
        # Input validation
        if not saved_model_name or saved_model_name.strip() == "":
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")

        if not query or query.strip() == "":
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logging.info(f'Received query: "{query}" for model: "{saved_model_name}"')

        # Call the prediction logic
        get_scope = f"Given query is {get_scope_for_query(query, saved_model_name)}"
        logging.info(f'Prediction for given is {get_scope}')

        return get_scope
    except Exception as exc:
        logging.error(f"Failed to get the scope for given query with error: {exc}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to get the scope for given query: {str(exc)}"})

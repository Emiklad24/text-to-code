from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


# Assuming your main processing logic is in a function `process_question` in your script.
from query import process_question  # Adjust import based on your script's name

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the request body
class QuestionRequest(BaseModel):
    question: str

# Define the response structure
class QueryResponse(BaseModel):
    query_string: str
    output_df: list[dict]  # We'll return the DataFrame as a dictionary
    question: str
    query: str


@app.post("/query", response_model=QueryResponse)
def get_query_result(request: QuestionRequest):
    try:
        # Process the question
        query_string, output_df, question, query = process_question(request.question)
        
        # Convert DataFrame to a list of dictionaries for JSON serialization
        output_df_dict = output_df.to_dict(orient='records')
        
        return QueryResponse(query_string=query_string, output_df=output_df_dict, question=question, query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

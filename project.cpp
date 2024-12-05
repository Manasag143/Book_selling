//Manas Agrawal
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct Book {
  string title;
  string author;
  bool isAvailable;
  double price; // Combined price for simplicity
};

int main() {
  vector<Book> books = {
    {"The Lord of the Rings", "J.R.R. Tolkien", true, 3.00},
    {"Pride and Prejudice", "Jane Austen", true, 2.00},
    {"The Hitchhiker's Guide to the Galaxy", "Douglas Adams", false, 4.00}
  };

  int choice, bookIndex;

  do {
    cout << "\n** Book Rental System **\n";
    cout << "1. View Available Books\n";
    cout << "2. Rent a Book\n";
    cout << "3. Return a Book\n";
    cout << "4. Exit\n";
    cout << "Enter your choice: ";
    cin >> choice;

    switch (choice) {
      case 1:
        cout << "\n** Available Books **\n";
        for (int i = 0; i < books.size(); ++i) {
          if (books[i].isAvailable) {
            cout << i + 1 << ". " << books[i].title << " by " << books[i].author << " ($" << books[i].price << ")" << endl;
          }
        }
        break;
      case 2: {
        cout << "\nEnter the number of the book you want to rent: ";
        cin >> bookIndex;

        if (bookIndex >= 1 && bookIndex <= books.size()) {
          if (books[bookIndex - 1].isAvailable) {
            books[bookIndex - 1].isAvailable = false;
            cout << "\n** Book Rented Successfully! **\n";
            cout << "Rental Price: $" << books[bookIndex - 1].price << endl;
          } else {
            cout << "\n** Book is not available! **\n";
          }
        } else {
          cout << "\n** Invalid book number! **\n";
        }
        break;
      }
      case 3: {
        cout << "\nEnter the number of the book you want to return: ";
        cin >> bookIndex;

        if (bookIndex >= 1 && bookIndex <= books.size()) {
          books[bookIndex - 1].isAvailable = true;
          cout << "\n** Book Returned Successfully! **\n";
        } else {
          cout << "\n** Invalid book number! **\n";
        }
        break;
      }
      case 4:
        cout << "\n** Exiting Book Rental System **\n";
        break;
      default:
        cout << "\n** Invalid choice! **\n";
    }
  } while (choice != 4);

  return 0;
}

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import logging
import uvicorn
from typing import Dict, Optional
from datetime import datetime

# Import your OLAP processing classes
from olap_processor import (
    OLAPQueryProcessor, 
    setup_logging
)

# Set up logging
logger = logging.getLogger('olap_genai')
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="OLAP Query Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    user_query: str
    cube_id: str

class QueryResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None
    dimensions: Optional[Dict] = None
    measures: Optional[Dict] = None
    processing_time: Optional[float] = None

# Initialize OLAP processor with configuration
config_path = "text2sql/config.json"
olap_processors = {}  # Dictionary to store processors for each user

# Token verification dependency
async def verify_token(request: Request):
    token = request.headers.get('authorization')
    if not token:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        token = token.split(" ")[1]
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("preferred_username")
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/genai/cube/query_generation", response_model=QueryResponse)
async def generate_cube_query(
    request: QueryRequest,
    username: str = Depends(verify_token)
):
    try:
        # Get or create processor for this user
        if username not in olap_processors:
            olap_processors[username] = OLAPQueryProcessor(config_path)
            logger.info(f"Created new OLAP processor for user: {username}")

        processor = olap_processors[username]

        # Process the query
        original_query, final_query, processing_time, dimensions, measures = processor.process_query(request.user_query)

        # Prepare response
        response = {
            "message": "success",
            "cube_query": final_query,
            "dimensions": dimensions,
            "measures": measures,
            "processing_time": processing_time
        }

        # Log the successful query
        logger.info(f"Successfully processed query for user {username}")
        logger.info(f"Query: {original_query}")
        logger.info(f"Generated OLAP query: {final_query}")

        return QueryResponse(**response)

    except Exception as e:
        logger.error(f"Error processing query for user {username}: {str(e)}")
        return QueryResponse(
            message="failure",
            cube_query=None,
            dimensions=None,
            measures=None,
            processing_time=None
        )

# Endpoint to clear conversation history
@app.post("/genai/cube/clear_history")
async def clear_history(username: str = Depends(verify_token)):
    try:
        if username in olap_processors:
            del olap_processors[username]
            logger.info(f"Cleared conversation history for user: {username}")
        return {"message": "success", "detail": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Error clearing history for user {username}: {str(e)}")
        return {"message": "failure", "detail": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    setup_logging()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8085,
        workers=4,
        log_level="info"
    )

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


from fastapi import APIRouter, FastAPI, Header, Request, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
import logging
import uuid
import json
import time
import threading
from threading import Thread
import os
import glob
from pydantic import BaseModel
from typing import Annotated, Dict, Optional, List
import jwt
from datetime import datetime

# Import your OLAP processor classes
from your_olap_module import OLAPQueryProcessor, QueryContext, ConversationManager

# Constants
DATA_FOLDER = "./datafolder/json/"
CONVERSATION_FOLDER = "./datafolder/conversations/"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CONVERSATION_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('olap_genai')

# Pydantic models
class CubeImportRequest(BaseModel):
    cube_json: dict
    cube_id: str

class QueryRequest(BaseModel):
    user_query: str
    cube_id: str

class QueryResponse(BaseModel):
    query: str
    dimensions: Dict
    measures: Dict
    olap_query: str
    timestamp: float
    conversation_id: str

# Global storage for user sessions
user_sessions: Dict[str, OLAPQueryProcessor] = {}
conversation_locks: Dict[str, threading.Lock] = {}

# FastAPI app and router
app = FastAPI(title="OLAP GenAI API")
router = APIRouter()

def get_user_from_token(token: str) -> str:
    """Extract and validate user from JWT token"""
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("preferred_username")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_conversation_file(user_id: str, cube_id: str) -> str:
    """Get the conversation file path for a user and cube"""
    return os.path.join(CONVERSATION_FOLDER, f"{user_id}_{cube_id}_conversation.json")

def load_conversation_history(user_id: str, cube_id: str) -> List[Dict]:
    """Load conversation history from file"""
    file_path = get_conversation_file(user_id, cube_id)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
    return []

def save_conversation_history(user_id: str, cube_id: str, conversation: Dict):
    """Save conversation history to file"""
    file_path = get_conversation_file(user_id, cube_id)
    try:
        with open(file_path, 'w') as f:
            json.dump(conversation, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

def get_or_create_processor(user_id: str, cube_id: str) -> OLAPQueryProcessor:
    """Get or create an OLAP processor for a user"""
    session_key = f"{user_id}_{cube_id}"
    if session_key not in user_sessions:
        config_path = os.path.join(DATA_FOLDER, f"{cube_id}_config.json")
        processor = OLAPQueryProcessor(config_path)
        user_sessions[session_key] = processor
        conversation_locks[session_key] = threading.Lock()
    return user_sessions[session_key]

@router.post("/genai/cube_details_import")
async def import_cube_details(
    request: Request,
    cube_data: CubeImportRequest
):
    """Import cube details and configuration"""
    token = request.headers.get('authorization', '').split(" ")[1]
    user_id = get_user_from_token(token)
    
    try:
        # Save cube configuration
        config_path = os.path.join(DATA_FOLDER, f"{cube_data.cube_id}_config.json")
        with open(config_path, 'w') as f:
            json.dump(cube_data.cube_json, f, indent=2)
            
        return JSONResponse(
            status_code=200,
            content={"message": "success", "cube_id": cube_data.cube_id}
        )
    except Exception as e:
        logger.error(f"Error importing cube details: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "failure", "error": str(e)}
        )

@router.post("/genai/query")
async def process_query(
    request: Request,
    query_request: QueryRequest
):
    """Process a query and maintain conversation history"""
    token = request.headers.get('authorization', '').split(" ")[1]
    user_id = get_user_from_token(token)
    
    session_key = f"{user_id}_{query_request.cube_id}"
    
    try:
        with conversation_locks[session_key]:
            processor = get_or_create_processor(user_id, query_request.cube_id)
            
            # Process the query
            query, olap_query, processing_time, dimensions, measures = processor.process_query(query_request.user_query)
            
            # Create response
            response = {
                "query": query,
                "dimensions": dimensions,
                "measures": measures,
                "olap_query": olap_query,
                "timestamp": time.time(),
                "conversation_id": str(uuid.uuid4()),
                "processing_time": processing_time
            }
            
            # Save conversation history
            conversation_history = load_conversation_history(user_id, query_request.cube_id)
            conversation_history.append(response)
            save_conversation_history(user_id, query_request.cube_id, conversation_history)
            
            return JSONResponse(
                status_code=200,
                content={"message": "success", "response": response}
            )
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "failure", "error": str(e)}
        )

@router.get("/genai/conversation_history")
async def get_conversation_history(
    request: Request,
    cube_id: str
):
    """Get conversation history for a user and cube"""
    token = request.headers.get('authorization', '').split(" ')[1]
    user_id = get_user_from_token(token)
    
    try:
        history = load_conversation_history(user_id, cube_id)
        return JSONResponse(
            status_code=200,
            content={"message": "success", "history": history}
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "failure", "error": str(e)}
        )

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8085,
        workers=4,
        log_level="info"
    )

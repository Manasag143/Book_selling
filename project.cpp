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

from fastapi import APIRouter, FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import jwt
from langchain_community.vectorstores.chroma import Chroma
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import logging
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
import asyncio
from pathlib import Path
import uvicorn
from cube_query_v3 import OLAPQueryProcessor
import shutil
from langchain_core.documents import Document
import subprocess

# Initialize FastAPI app
app = FastAPI(title="OLAP Cube Management API")
router = APIRouter()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    user_query: str
    cube_id: int

class QueryResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None

class CubeErrorRequest(BaseModel):
    user_query: str
    cube_id: str
    error_message: str

class CubeErrorResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None
    error_details: Optional[Dict] = None

class CubeDetailsRequest(BaseModel):
    cube_json_dim: List[Dict]
    cube_json_msr: List[Dict]
    cube_id: str

class CubeDetailsResponse(BaseModel):
    message: str

# Configuration and storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
ERROR_HISTORY_FILE = os.path.join(BASE_DIR, "error_history.json")
history_file = "conversation_history.json"
vector_db_path = "vector_db"
config_file = "text2sql/config.json"

# Initialize logging with error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    filename='api_with_errors.log'
)

class ErrorHistory:
    """Tracks error history for queries"""
    def __init__(self, history_file: str = ERROR_HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading error history: {e}")
            return {}

    def save(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving error history: {e}")

    def add_error(self, user_id: str, query: str, error: str, correction: str):
        if user_id not in self.history:
            self.history[user_id] = []
        
        self.history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": error,
            "correction": correction
        })
        self.save()

class LLMConfigure:
    """Configuration for LLM with error handling"""
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None
        self.error_history = ErrorHistory()

    # ... (rest of the LLMConfigure class remains the same)

# Token verification with error tracking
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, options={"verify_signature": False})
        user_details = payload.get("preferred_username")
        if not user_details:
            raise ValueError("No user details in token")
        return user_details
    except Exception as e:
        logging.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail=str(e))

# Initialize OLAP processor dictionary
olap_processors = {}

@app.post("/genai/cube_error_injection", response_model=CubeErrorResponse)
async def handle_cube_error(request: CubeErrorRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        
        if user_id not in olap_processors:
            olap_processors[user_id] = OLAPQueryProcessor(config_file)
        
        processor = olap_processors[user_id]
        history_manager = History()
        error_history = ErrorHistory()
        prev_conversation = history_manager.retrieve(user_id)
        
        query, final_query, processing_time, dimensions, measures = processor.process_query_with_error(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.error_message
        )
        
        # Record error and correction
        error_history.add_error(
            user_id,
            request.user_query,
            request.error_message,
            final_query
        )
        
        response_data = {
            "query": query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,
        }
        
        history_manager.update(user_id, response_data)
        
        return CubeErrorResponse(
            message="success",
            cube_query=final_query,
            error_details={
                "original_error": request.error_message,
                "correction_timestamp": datetime.now().isoformat()
            }
        )
    
    except HTTPException as he:
        logging.error(f"HTTP Exception in handle_cube_error: {he}")
        return CubeErrorResponse(
            message="failure",
            cube_query=None,
            error_details={"error_type": "http_error", "details": str(he)}
        )
    except Exception as e:
        logging.error(f"Error in handle_cube_error: {e}")
        return CubeErrorResponse(
            message="failure",
            cube_query=None,
            error_details={"error_type": "processing_error", "details": str(e)}
        )

@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: QueryRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        result = await process_query(request.user_query, request.cube_id, user_id)
        return QueryResponse(
            message=result["message"],
            cube_query=result["cube_query"]
        )
    except HTTPException as he:
        return QueryResponse(message="failure", cube_query=None)
    except Exception as e:
        logging.error(f"Error in generate_cube_query: {e}")
        return QueryResponse(message="failure", cube_query=None)

@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(request: CubeDetailsRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        result = await process_cube_details(
            request.cube_json_dim,
            request.cube_json_msr,
            request.cube_id
        )
        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        return CubeDetailsResponse(message="failure")
    except Exception as e:
        logging.error(f"Error in import_cube_details: {e}")
        return CubeDetailsResponse(message="failure")

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        # Create required directories
        for directory in [CUBE_DETAILS_DIR, vector_db_path]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize history files
        for file in [IMPORT_HISTORY_FILE, history_file, ERROR_HISTORY_FILE]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump({}, f)
        
        logging.info("API startup completed successfully with error handling")
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        raise

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("olap_details_generat:app", host="127.0.0.1", port=8085, reload=True, workers=4)

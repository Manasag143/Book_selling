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
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import jwt
from langchain_chroma import Chroma
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

# Initialize FastAPI app
app = FastAPI(title="OLAP Cube Management API")

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

class CubeDetailsRequest(BaseModel):
    cube_json: Dict
    cube_id: str

class CubeDetailsResponse(BaseModel):
    message: str

# Configuration and storage paths
BASE_DIR = os.getcwd()  # Get current working directory
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
history_file = os.path.join(BASE_DIR, "conversation_history.json")
vector_db_path = os.path.join(BASE_DIR, "vector_db")
config_file = os.path.join(BASE_DIR, "text2sql/config.json")

# Load Azure Configuration
with open(config_file, 'r') as f:
    config = json.load(f)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = config["llm"]["OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = config["llm"]["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = config["llm"]["DEPLOYMENT_NAME"]
AZURE_API_VERSION = config["llm"]["OPENAI_API_VERSION"]
EMBEDDING_DEPLOYMENT = config["embedding"]["deployment"]

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='api.log'
)

class VectorStore:
    def __init__(self, persist_directory: str):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_cube_details(self, cube_id: str, cube_json: Dict):
        # Convert cube details to text format for embedding
        cube_text = json.dumps(cube_json, indent=2)
        metadata = {"cube_id": cube_id, "type": "cube_details"}
        
        # Add to vector store
        self.vectorstore.add_texts(
            texts=[cube_text],
            metadatas=[metadata]
        )
        self.vectorstore.persist()

    def search_similar_cubes(self, query: str, k: int = 3):
        return self.vectorstore.similarity_search(query, k=k)

class History:
    def __init__(self, history_file: str = history_file):
        self.history_file = history_file
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading conversation history: {e}")
            return {}

    def save(self, history: Dict):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving conversation history: {e}")

    def update(self, user_id: str, query_data: Dict):
        if user_id not in self.history:
            self.history[user_id] = []

        self.history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "query": query_data["query"],
            "dimensions": query_data["dimensions"],
            "measures": query_data["measures"],
            "response": query_data["response"]
        })
        self.history[user_id] = self.history[user_id][-5:]
        self.save(self.history)

class ImportHistory:
    def __init__(self, history_file: str = IMPORT_HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading import history: {e}")
            return {}
    
    def save(self, history: Dict):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving import history: {e}")

    def update(self, user_id: str, cube_id: str, status: str):
        if user_id not in self.history:
            self.history[user_id] = []

        new_import = {
            "timestamp": datetime.now().isoformat(),
            "cube_id": cube_id,
            "status": status
        }

        self.history[user_id].append(new_import)
        self.history[user_id] = self.history[user_id][-5:]
        self.save(self.history)

# Token verification
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
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize OLAP processor dictionary
olap_processors = {}

async def process_query(user_query: str, cube_id: int, user_id: str) -> Dict:
    try:
        # Get or create processor for this user
        if user_id not in olap_processors:
            olap_processors[user_id] = OLAPQueryProcessor(config_file)

        processor = olap_processors[user_id]
        
        # Search for similar patterns in both dimensions and measures
        similar_dimensions = app.state.dimensions_vectorstore.search_similar_cubes(user_query)
        similar_measures = app.state.measures_vectorstore.search_similar_cubes(user_query)
        
        # Combine the results
        similar_cubes = similar_dimensions + similar_measures
        
        # Process the query with context from similar cubes
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            user_query,
            context=similar_cubes
        )
        
        # Prepare response data
        response_data = {
            "query": query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,
        }

        # Update conversation history
        history_manager = History()
        history_manager.update(user_id, response_data)

        return {
            "message": "success",
            "cube_query": final_query
        }
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return {
            "message": "failure",
            "cube_query": None
        }

async def process_cube_details(cube_json: Dict, cube_id: str, user_id: str) -> Dict:
    try:
        # Save cube details to file
        cube_dir = os.path.join(CUBE_DETAILS_DIR, cube_id)
        os.makedirs(cube_dir, exist_ok=True)

        cube_file = os.path.join(cube_dir, f"cube_details.json")
        with open(cube_file, 'w') as f:
            json.dump(cube_json, f, indent=4)

        # Separate dimensions and measures from cube_json
        dimensions = cube_json.get('dimensions', {})
        measures = cube_json.get('measures', {})

        # Add to respective vector stores
        if dimensions:
            app.state.dimensions_vectorstore.add_cube_details(cube_id, dimensions)
        if measures:
            app.state.measures_vectorstore.add_cube_details(cube_id, measures)

        # Update import history
        history_manager = ImportHistory()
        history_manager.update(user_id, cube_id, "success")

        return {"message": "success"}
    except Exception as e:
        logging.error(f"Error processing cube details: {e}")
        return {"message": "failure"}

@app.post("/genai/cube/query_generation", response_model=QueryResponse)
async def generate_cube_query(
    request: QueryRequest,
    user_details: str = Depends(verify_token)
):
    try:
        user_id = f"user_{user_details}"
        result = await process_query(
            request.user_query,
            request.cube_id,
            user_id
        )
        return QueryResponse(
            message=result["message"],
            cube_query=result["cube_query"]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in generate_cube_query: {e}")
        return QueryResponse(message="failure", cube_query=None)

@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(
    request: CubeDetailsRequest,
    user_details: str = Depends(verify_token)
):
    try:
        user_id = f"user_{user_details}"
        result = await process_cube_details(
            request.cube_json,
            request.cube_id,
            user_id
        )
        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in import_cube_details: {e}")
        return CubeDetailsResponse(message="failure")

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        # Create necessary directories using absolute paths
        os.makedirs(CUBE_DETAILS_DIR, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        # No need to create directory for IMPORT_HISTORY_FILE since we're using BASE_DIR

        # Initialize history files
        for file in [IMPORT_HISTORY_FILE, history_file]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump({}, f)

        # Initialize vector store using paths from config
        dimensions_path = config["vector_embedding_path"]["dimensions"]
        measures_path = config["vector_embedding_path"]["measures"]
        
        # Create vector store directories if they don't exist
        os.makedirs(dimensions_path, exist_ok=True)
        os.makedirs(measures_path, exist_ok=True)
        
        # Initialize vector stores for dimensions and measures
        app.state.dimensions_vectorstore = VectorStore(dimensions_path)
        app.state.measures_vectorstore = VectorStore(measures_path)

        # Initialize Azure OpenAI chat model
        app.state.llm = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY
        )

        logging.info("API startup completed successfully")
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run("olap_details_generat:app", host="127.0.0.1", port=8085, reload=True)

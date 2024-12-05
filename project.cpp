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
from pydantic import BaseModel
from typing import Dict, Optional
import json
import os
from datetime import datetime
import logging
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Cube Details Import API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CubeDetailsRequest(BaseModel):
    cube_json: Dict
    cube_id: str

class CubeDetailsResponse(BaseModel):
    message: str

# Configuration and storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cube_import.log'
)

class ImportHistory:
    def __init__(self, history_file: str = IMPORT_HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load()
        print(f"Import history file location: {self.history_file}")

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded import history: {data}")
                    return data
            print(f"No existing import history file, creating new one")
            return {}
        except Exception as e:
            logging.error(f"Error loading import history: {e}")
            print(f"Error loading import history: {e}")
            return {}
    
    def save(self, history: Dict):
        try:
            print(f"Saving import history: {history}")
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Import history saved successfully")
        except Exception as e:
            logging.error(f"Error saving import history: {e}")
            print(f"Error saving import history: {e}")

    def update(self, user_id: str, cube_id: str, status: str):
        print(f"Updating import history for user {user_id}")
        if user_id not in self.history:
            self.history[user_id] = []

        new_import = {
            "timestamp": datetime.now().isoformat(),
            "cube_id": cube_id,
            "status": status
        }

        print(f"Adding import record: {new_import}")
        self.history[user_id].append(new_import)
        self.history[user_id] = self.history[user_id][-5:]  # Keep last 5 imports
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

async def process_cube_details(
    cube_json: Dict,
    cube_id: str,
    user_id: str
) -> Dict:
    try:
        # Create directory for cube details if it doesn't exist
        cube_dir = os.path.join(CUBE_DETAILS_DIR, cube_id)
        os.makedirs(cube_dir, exist_ok=True)

        # Save cube details to file
        cube_file = os.path.join(cube_dir, f"cube_details.json")
        with open(cube_file, 'w') as f:
            json.dump(cube_json, f, indent=4)

        # Update import history
        history_manager = ImportHistory()
        history_manager.update(user_id, cube_id, "success")

        return {"message": "success"}
    except Exception as e:
        logging.error(f"Error processing cube details: {e}")
        return {"message": "failure"}

@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(
    request: CubeDetailsRequest,
    user_details: str = Depends(verify_token)
):
    try:
        user_id = f"user_{user_details}"

        # Process cube details
        result = await process_cube_details(
            request.cube_json,
            request.cube_id,
            user_id
        )

        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        return CubeDetailsResponse(message="failure")
    except Exception as e:
        logging.error(f"Error in import_cube_details: {e}")
        return CubeDetailsResponse(message="failure")

# History retrieval endpoint
@app.get("/genai/cube_details/history/{user_id}")
async def get_import_history(
    user_id: str,
    user_details: str = Depends(verify_token)
):
    history_manager = ImportHistory()
    user_history = history_manager.history.get(user_id, [])
    return {"history": user_history}

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        # Ensure directories exist
        os.makedirs(CUBE_DETAILS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(IMPORT_HISTORY_FILE), exist_ok=True)

        # Create history file if it doesn't exist
        if not os.path.exists(IMPORT_HISTORY_FILE):
            with open(IMPORT_HISTORY_FILE, 'w') as f:
                json.dump({}, f)
            print(f"Created new import history file at {IMPORT_HISTORY_FILE}")

        # Test write permissions
        with open(IMPORT_HISTORY_FILE, 'a') as f:
            pass

        logging.info("API startup completed successfully")
        print(f"Import history file is ready at {IMPORT_HISTORY_FILE}")
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        print(f"Startup error: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

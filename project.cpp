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
}import logging
import time
import json
import pandas as pd
import csv
from typing import Dict, List, Tuple, Optional
import os
import requests
from datetime import date
from colorama import Fore, Style, init
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
ERROR_HISTORY_FILE = os.path.join(BASE_DIR, "error_history.json")
history_file = "conversation_history.json"
vector_db_path = "vector_db"
config_file = "config.json"

def setup_logging():
    """Enhanced logging setup with error tracking"""
    today = date.today()
    log_folder = './log'
    error_folder = './log/errors'
  
    for folder in [log_folder, error_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Main logging
    logging.basicConfig(
        filename=f"{log_folder}/{today}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    # Error logging
    error_handler = logging.FileHandler(f"{error_folder}/{today}_errors.log")
    error_handler.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(error_handler)

class DimensionMeasure:
    """Enhanced dimension and measure extraction with error handling"""
    def __init__(self, llm: str, embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 5

    def get_dimensions_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
        """Extracts dimensions with error correction"""
        try:
            with get_openai_callback() as dim_cb:
                query_dim_error_inj = f"""
                As a user query to cube query conversion expert, analyze the error message from applying the cube query. 
                Your goal is to correct the identification of dimensions if they are incorrect.
                
                Below details using which the error occurred:-
                User Query: {query}
                Current Dimensions: {prev_conv["dimensions"]}
                Current Cube Query Response: {prev_conv["response"]}
                Error Message: {error}

                <error_analysis_context>
                - Analyze error message for specific dimension issues
                - Check for syntax errors in dimension references
                - Identify missing or incorrect dimension hierarchies
                - Look for dimension name mismatches
                - Consider temporal context and aggregation levels
                </error_analysis_context>

                <correction_instructions>
                - Identify and fix dimension name mismatches
                - Correct hierarchy level references
                - Handle missing dimensions appropriately
                - Maintain valid dimension relationships
                - Preserve correct temporal contexts
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["corrected_group1", "corrected_group2"],
                    "dimension_level_names": ["corrected_level1", "corrected_level2"],
                    "corrections": ["Changed X to Y", "Added missing dimension Z"],
                    "error_analysis": "Description of what was wrong and why"
                }}
                '''
                </correction_instructions>
                """

                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                
                persist_directory_dim = cube_dim
                load_embedding_dim = Chroma(
                    persist_directory=persist_directory_dim,
                    embedding_function=self.embedding
                )
                
                retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 50})
                
                chain_dim = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_dim,
                    verbose=True,
                    return_source_documents=True
                )
                
                result_dim = chain_dim.invoke({"query": query_dim_error_inj})
                dim = result_dim.get('result')
                
                logging.info(f"Dimension correction results: {dim}")
                logging.info(f"Token usage - Total: {dim_cb.total_tokens}, Cost: ${dim_cb.total_cost}")
                
                return dim

        except Exception as e:
            logging.error(f"Errorponse:{prev_conv["response"]}
                Error Message:- {error}

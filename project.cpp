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
import logging
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
history_file = "conversation_history.json"
vector_db_path = "vector_db"
config_file = "config.json"

def setup_logging():
    """Setup logging configuration"""
    today = date.today()
    log_folder = './log'
  
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    logging.basicConfig(
        filename=f"{log_folder}/{today}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class LLMConfigure:
    """Class responsible for loading and configuring LLM and embedding models."""
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                logging.info("Config file loaded successfully.")
                return config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise

    def initialize_llm(self):
        try:
            self.llm = AzureChatOpenAI(
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                model=self.config['llm']['model'],
                temperature=self.config['llm']['temperature'],
                api_version=self.config['llm']["OPENAI_API_VERSION"],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                seed=self.config['llm']["seed"]
            )
            return self.llm
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            raise

    def initialize_embedding(self):
        try:
            self.embedding = AzureOpenAIEmbeddings(
                deployment=self.config['embedding']['deployment'],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                show_progress_bar=self.config['embedding']['show_progress_bar'],
                disallowed_special=(),
                openai_api_type=self.config['llm']['OPENAI_API_TYPE']
            )
            return self.embedding
        except Exception as e:
            logging.error(f"Error initializing embedding: {e}")
            raise

class DimensionMeasure:
    """Class responsible for extracting dimensions and measures with error handling."""
    def __init__(self, llm: str, embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 5

    def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extract dimensions from query."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim = f"""
                As an OLAP query expert, analyze the user's question and identify required dimension information.
                
                User Query: {query}
                Previous Query: {prev_conv["query"]}
                Previous Dimensions: {prev_conv["dimensions"]}
                Previous Query Response: {prev_conv["response"]}

                <context>
                - Consider if this is a follow-up question
                - Look for temporal references
                - Check for comparative references
                - Identify implicit dimensions from context
                </context>
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["group1", "group2"],
                    "dimension_level_names": ["level1", "level2"]
                }}
                '''
                """
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                load_embedding_dim = Chroma(
                    persist_directory=cube_dim,
                    embedding_function=self.embedding
                )
                retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 50})
                chain_dim = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_dim,
                    verbose=True,
                    return_source_documents=True
                )
                result_dim = chain_dim.invoke({"query": query_dim})
                return result_dim.get('result')
        except Exception as e:
            logging.error(f"Error extracting dimensions: {e}")
            raise

    def get_dimensions_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
        """Extract dimensions with error correction."""
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
                - Analyze the error message to identify incorrect dimension references
                - Check for syntax errors in dimension names/hierarchy levels
                - Verify dimension existence in cube structure
                - Identify missing or invalid dimension combinations
                - Consider temporal context for date/time dimensions
                </error_analysis_context>

                <correction_guidelines>
                - Fix incorrect dimension names
                - Correct hierarchy references
                - Add required missing dimensions
                - Remove invalid combinations
                - Preserve valid dimensions
                - Maintain temporal relationships
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["corrected_group1", "corrected_group2"],
                    "dimension_level_names": ["corrected_level1", "corrected_level2"],
                    "corrections": ["Fixed dimension X", "Added missing dimension Y"],
                    "reasoning": "Explanation of corrections made"
                }}
                '''
                </correction_guidelines>

                <examples>
                1. Error: "Unknown dimension [Time].[Date]"
                Correction: Change to [Time].[Month] since Date level doesn't exist

                2. Error: "Invalid hierarchy reference [Geography].[City]"
                Correction: Change to [Geography].[Region].[City] to include parent

                3. Error: "Missing required dimension [Product]"
                Correction: Add [Product].[Category] based on context

                4. Error: "Incompatible dimensions [Customer] and [Item]"
                Correction: Remove [Item] dimension as it's incompatible
                </examples>
                """
                
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                load_embedding_dim = Chroma(
                    persist_directory=cube_dim,
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
                return result_dim.get('result')

        except Exception as e:
            logging.error(f"Error in dimension correction: {e}")
            raise

    def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extract measures from query."""
        try:
            with get_openai_callback() as msr_cb:
                query_msr = f"""
                As an OLAP query expert, analyze the user's question and identify required measures.
                
                User Query: {query}
                Previous Query: {prev_conv["query"]}
                Previous Measures: {prev_conv["measures"]}
                Previous Query Response: {prev_conv["response"]}

                <context>
                - Consider follow-up questions referencing previous measures
                - Look for comparative phrases
                - Check for aggregation references
                - Identify implicit measures from context
                </context>
                
                Response format:
                '''json
                {{
                    "measure_group_names": ["group1", "group2"],
                    "measure_names": ["measure1", "measure2"]
                }}
                '''
                """
                
                cube_msr = os.path.join(vector_db_path, cube_id, "measures")
                load_embedding_msr = Chroma(
                    persist_directory=cube_msr,
                    embedding_function=self.embedding
                )
                retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 60})
                chain_msr = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_msr,
                    verbose=True,
                    return_source_documents=True
                )
                result_msr = chain_msr.invoke({"query": query_msr})
                return result_msr.get('result')

        except Exception as e:
            logging.error(f"Error extracting measures: {e}")
            raise

class OLAPQueryProcessor:
    """OLAP query processor with error handling."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.conversation_manager = ConversationManager()
        self.memory = ConversationBufferMemory(return_messages=True)

        try:
            self.llm_config = LLMConfigure(config_path)
            self.load_json = self.llm_config.load_config(config_path)
            self.llm = self.llm_config.initialize_llm()
            self.embedding = self.llm_config.initialize_embedding()
            self.dim_measure = DimensionMeasure(self.llm, self.embedding, self.load_json)
            self.query_history = []
        except Exception as e:
            logging.error(f"Error initializing OLAPQueryProcessor: {e}")
            raise

    def process_query(self, query: str, cube_id: str, prev_conv: dict) -> Tuple[str, str, float, Dict, Dict]:
        """Process a regular query."""
        try:
            start_time = time.time()
            dimensions = self.dim_measure.get_dimensions(query, cube_id, prev_conv)
            measures = self.dim_measure.get_measures(query, cube_id, prev_conv)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures")

            final_query = self.generate_query(query, dimensions, measures, prev_conv)
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures

        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            raise

    def process_query_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> Tuple[str, str, float, Dict, Dict]:
        """Process a query with error correction."""
        try:
            start_time = time.time()
            
            # Get corrected dimensions and measures
            dimensions = self.dim_measure.get_dimensions_with_error(query, cube_id, prev_conv, error)
            measures = self.dim_measure.get_measures(query, cube_id, prev_conv)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures after error correction")

            final_query = self.generate_query(query, dimensions, measures, prev_conv)
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures

        except Exception as e:
            logging.error(f"Error in query processing with error correction: {e}")
            raise

    def generate_query(self, query: str, dimensions: Dict, measures: Dict, prev_conv: Dict) -> str:
        """Generate the final OLAP query."""
        try:
            generation_prompt = f"""
            Generate an OLAP query based on:
            User Query: {query}
            Dimensions: {dimensions}
            Measures: {measures}
            Previous Context: {prev_conv}

            Requirements:
            1. Use proper dimension and measure names
            2. Follow OLAP syntax rules
            3. Include appropriate aggregations
            4. Handle temporal context
            5. Format according to standards
            """

            result = self.llm.invoke(generation_prompt)
            return result.content

        except Exception as e:
            logging.error(f"Error generating query: {e}")
            raise

def main():
    """Main function with error handling."""
    setup_logging()
    try:
        processor = OLAPQueryProcessor(config_file)
        print(Fore.CYAN + "\n=== OLAP Query Processing System ===")
        print(Fore.CYAN + "Type 'exit' to end.\n")
        
        while True:
            try:
                query = input(Fore.GREEN + "Enter query: ")
                if query.lower() == 'exit':
                    print(Fore.YELLOW + "\nGoodbye!")
                    break
                
                result = processor.process_query(query)
                print(Fore.CYAN + f"\nProcessed successfully!")
                print(Fore.WHITE + f"Result: {result}\n")
                
            except Exception as e:
                print(Fore.RED + f"\nError: {str(e)}")
                print(Fore.YELLOW + "Please try again.\n")
                continue

    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(Fore.RED + "Critical error occurred. Check logs.")

if __name__ == "__main__":
    main()

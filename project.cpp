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
from logging import config
from fastapi import FastAPI,HTTPException,Header,Depends
import jwt
from langchain_chroma import Chroma
from pydantic import BaseModel, ConfigDict
from typing import Dict, List,Optional
import json
import os
from datetime import datetime
import logging
from langchain_community.vectorstores import chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
import asyncio
from pathlib import Path
from langchain_community.callbacks import get_openai_callback
import requests
import warnings
import uvicorn
from pydantic import BaseModel
from config import Config
from cube_query_v3 import OLAPQueryProcessor

#initialize FastAPI app
app = FastAPI(titkle="Cube Query Generation API")

#py
class QueryRequest(BaseModel):
    token: str
    user_query: str
    cube_id:int

class QueryResponse(BaseModel):
    message:str
    cube_query: str
    token_info: Optional[List] = None


#Configuration and storage path

history_file = "conversation_history.json"
vector_db_path = "vector_db"
config_file = "text2sql\config.json"

#Initialize logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s- %(message)s',
    filename = 'api.log'
)
class History:

    def __init__(self, history_file: str = history_file):
        self.history_file = history_file
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return{}
        except Exception as e: 
            logging.error(f"Error loading conversation history: {e}")
            return {}
    
    def save(self, history: Dict):
        #save conversation history to file
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e: 
            logging.error(f"Error saving conversation history: {e}")

    def update(self, user_id:str, query_data: Dict):
    #Update history for a specific user
        if user_id not in self.history:
            self.history[user_id]=[]

        #Add new conversation
        self.history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "query": query_data["query"],
            "dimensions": query_data["dimensions"],
            "measures" : query_data["measures"],
            "response" : query_data["response"]
        })
        #keep 5 last conversation
        self.history[user_id]= self.history[user_id][-5:]
        self.save(self.history)

def verify_and_decode_token(token:str)->tuple[List, str]:
    """
    verify and decode JWT token
    Return tuplr of (token info list, user details)
    """
    try:
         payload = jwt.decode(token, options={"verify_signature": False})
         print(f"Decoded token payload: {payload}")

         #extract user details
        #  user_details = payload.get("preferred_username")
         user_details = payload.get("Username")
         if not user_details:
             raise ValueError("No user details in token ")
         
         #create info message 
         info={"message": f"Hello,{user_details}!"}

         #create list with token and info
         mylist = []
         mylist.append(token)
         mylist.append(info)
         return mylist, user_details
    
    except Exception as e: 
        logging.error(f"Token verification failed: {e}") 
        raise HTTPException(status_code=401, detail={"message":"Invalid tokens"})      

async def process_query(
        user_query: str,
        cube_id: int,
        user_id:str) -> Dict:
    #process the user query and generate cube query
    try:
        processor = OLAPQueryProcessor(config_file)
        query, final_query,_,dimensions,measures =processor.process_query(user_query)
        #prepare response data
        response_data={
            "query": query,
            "dimensions": dimensions,
            "measures":measures,
            "response":final_query,
        }
        #update conversatio history
        history_manager = History()
        history_manager.update(user_id, response_data)

        return{
            "message":"success",
            "cube_query": final_query
        }
    except Exception as e: 
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
    
@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: QueryRequest):
    """
    Generate cube query from natural language input with JWT token handling
    """

    print("Input :",request)
    
    try:
        token_info = None
        user_id = None

        #handle token if present
        if request.token:
            token_info, user_details =verify_and_decode_token(request.token)
            user_id = f"user_{user_details}"
        else:
            #generate temporary user id if no token 
            user_id= f"user_{int(datetime.now().timestamp())}"

        #Load configuration
        #config = config(config_file)
        #print("user_id to check workinggggggggggggggggggggggggggggggggggggggggggggggggggggg",config)


        #process query
        result = await process_query(
            request.user_query,
            request.cube_id,
            user_id
            #config
        ) 

        return QueryResponse(
            message="success",
            cube_query=result["cube_query"],
            token_info=token_info
        )
    
    except HTTPException:
        return
    except Exception as e:
        logging.error(f"Error in generate_cube_query:{e}")
        return HTTPException(status_code=500, detail=(e))
        
 #startup event to ensure required directories exist 
@app.on_event("startup")
async def startup_event():
    """
    Create necessary directory and files on startup
    """
    try:
        Path(vector_db_path).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(history_file):
            with open(history_file, 'w') as f:
                json.dump({}, f)

        logging.info("API startup completed successfully")
    except Exception as e:
        logging.error(f"Error during startup:{e}")
        raise
    
if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)















  
import logging
import time
import json
import pandas as pd
import csv
from typing import Dict, List, Tuple
import os
import requests
from datetime import date
from colorama import Fore, Style ,init
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback


def setup_logging():
  today = date.today()
  log_folder = './log'
  
  if not os.path.exists(log_folder):
    os.mkdir(log_folder)
    
  logging.basicConfig(filename = f"{log_folder}/{today}.log",level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')


class LLMConfigure:
    """
    Class responsible for loading and configuring LLM and embedding models from a config file.
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                logging.info("Config file loaded successfully.")
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise

    def initialize_llm(self):
        """Initializes and returns the LLM model."""
        try:
            # Simulate LLM initialization using the config
            #self.llm = self.config['llm']
            self.llm = AzureChatOpenAI(openai_api_key= self.config['llm']['OPENAI_API_KEY'],
                                      model=self.config['llm']['model'],
                                      temperature=self.config['llm']['temperature'],
                                      api_version= self.config['llm']["OPENAI_API_VERSION"],
                                      azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      seed=self.config['llm']["seed"]
            )
            logging.info(f"LLM model initialized: {self.llm}")
            return self.llm
        except KeyError as e:
            logging.error(f"Missing LLM configuration in config file: {e}")
            raise

    def initialize_embedding(self):
        """Initializes and returns the Embedding model."""
        try:
            #embedding initialization using the config
            self.embedding = AzureOpenAIEmbeddings(deployment = self.config['embedding']['deployment'],
                                      azure_endpoint = self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      openai_api_key = self.config['llm']['OPENAI_API_KEY'],
                                      show_progress_bar = self.config['embedding']['show_progress_bar'],
                                      disallowed_special = (),
                                      openai_api_type = self.config['llm']['OPENAI_API_TYPE']
                          
                        )
            logging.info(f"Embedding model initialized: {self.embedding}")
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise

class DimensionMeasure:
    """
    Class responsible for extracting dimensions and measures from the natural language query.
    """

    def __init__(self, dimensions: str, measures: str,llm: str,embedding: str):
        self.dim = dimensions
        self.msr = measures
        self.llm = llm
        self.embedding = embedding

    def create_dimensions_db(self, cube_id):
        text_list_dim = self.dim.split("\n")
        text_list_dim = [Document(i) for i in text_list_dim]
        
        persist_directory_dimensions = f"./vectordb/{cube_id}/dimensions"

        if not os.path.exists(persist_directory_dimensions):
            os.mkdir(persist_directory_dimensions)

        vectordb = Chroma.from_documents(
        documents=text_list_dim , # chunks
        embedding=self.embedding, # instantiated embedding model
        persist_directory=persist_directory_dimensions # directory to save the data
        )
    
    def create_measures_db(self, cube_id):
        text_list_msr = self.msr.split("\n")
        text_list_msr = [Document(i) for i in text_list_msr]
        
        persist_directory_measures = f"./vectordb/{cube_id}/measures"

        if not os.path.exists(persist_directory_measures):
            os.mkdir(persist_directory_measures)

        vectordb = Chroma.from_documents(
        documents=text_list_msr , # chunks
        embedding=self.embedding, # instantiated embedding model
        persist_directory=persist_directory_measures # directory to save the data
        )

class OLAPQueryProcessor:
    """
    Main class for processing OLAP queries from natural language or Excel inputs.
    """

    def __init__(self,query: str, config_path: str,groundtruth: str):
        self.query = query
        self.llm_config = LLMConfigure(config_path)
        self.llm = self.llm_config.initialize_llm()
        self.embedding = self.llm_config.initialize_embedding()
        
    
    def process_query(self, cube_json, cube_id):

        self.dimensions = """Group Name:Risk Quality--Level Name:Asset Classification--Description:Asset Classification of the account like SMA1, Doubtful, NPA etc.
Group Name:Risk Quality--Level Name:Asset Type--Description:Asset Type
Group Name:TDS Details--Level Name:TDS Nature  Payment--Description:Tax Deducted at Source - Nature of Payment
Group Name:TDS Details--Level Name:TDS Head Code--Description:Tax Deducted at Source - Head Codes like company deductees, non-company deductees
Group Name:Contract Details--Level Name:Std Product Code--Description:Standard Product Code as maintained internally by the bank
Group Name:Contract Details--Level Name:Contract Customer Code--Description:Contract Customer Code as maintained internall by the bank
Group Name:GL Details--Level Name:GL No--Description:General Ledger Number
Group Name:GL Details--Level Name:GL Description--Description:General Ledger Description 
Group Name:GL Details--Level Name:GL Currency--Description:General Ledger Currency
Group Name:Benchmark Index Details--Level Name:Index Name--Description:Benchmark Index Name like BSE 500, BSE Bankex etc.
Group Name:Statutary Dues--Level Name:Nature Of Dues--Description:Nature of Statutary Dues like Customs, Excise etc.
Group Name:GST Filing--Level Name:GST Filing Type--Description:Goods and Services Tax (GST) Filing Type like GSTR1, GSTR10, GST3B etc.
Group Name:GST Filing--Level Name:GST  Filing Status--Description:Goods and Services Tax (GST) Filing Status - Filed or Not Filed
Group Name:Bulk Deal Details--Level Name:Security Name--Description:Bulk Deal Details - Name of the security / company on which bulk deals have happened in securities market
Group Name:Bulk Deal Details--Level Name:Client Name--Description:Name of the client - fund house/investment companies etc. doing the bulk deals
Group Name:Bulk Deal Details--Level Name:Buy-Sell--Description:Flag that indicates whether it is a Buy or Sell transaction in bulk deal by the client.
"Group Name:Mutual Fund Investment--Level Name:ISIN Number--Description:ISIN Number which is a unique identification of financial  instruments, including equity, debt etc. as per the International Securities Identification Numbering System Indicates the security / ISIN number on which a mutual fund has traded"
Group Name:Mutual Fund Investment--Level Name:Scheme Code--Description:Scheme Code of the Mutual Fund - Unique code given to every Mutual Fund 
Group Name:Mutual Fund Investment--Level Name:Mutual Fund Name--Description:Mutual Fund Name
Group Name:Mutual Fund Investment--Level Name:Scheme  Name--Description:Scheme Name of the Mutual Fund
Group Name:Mutual Fund Investment--Level Name:Rating--Description:Rating of the Debt Instrument
Group Name:Mutual Fund Investment--Level Name:Security Class--Description:Security Class like Commerical Papers (CP), Non-convertible debentures NCDs and Bonds etc.
Group Name:Branch Details--Level Name:City--Description:Branch City 
Group Name:Branch Details--Level Name:BSR Code 2--Description:BSR Code 2  - Unique code given to branch 
Group Name:Branch Details--Level Name:State--Description:State in which the branch is located
Group Name:Branch Details--Level Name:Branch Name--Description:Name of the branch
Group Name:Branch Details--Level Name:BSR Code 1--Description:BSR Code 1 - Unique code given to branch as per RBI
Group Name:Branch Details--Level Name:Branch Code--Description:Branch Code -  - Unique code given to branch by bank internally
Group Name:Branch Details--Level Name:Zone--Description:Zone in which branch is located - North, South, East etc.
Group Name:DRT Details--Level Name:DRT Case Type--Description:Debt Recovery Tribunal (DRT) Case Type like Regular Appeal, Transfer Application, SARFAESI Application etc.
"Group Name:Financial Yeild--Level Name:ISIN--Description:ISIN Number which is a unique identification of financial  instruments, including equity, debt etc. as per the International Securities Identification Numbering System Indicates the security / ISIN number on which finanical yield movements can be tracked"
Group Name:Limit Expiry Date--Level Name:DateSk--Description:Limit Expiry Date in YYYYMMDD format
Group Name:Customer Details--Level Name:Customer code--Description:Customer code - Unique identifier for each customer in the bank
Group Name:Customer Details--Level Name:Customer ID--Description:Customer ID - Unique identifier for each customer in the bank
Group Name:Customer Details--Level Name:Customer Name--Description:Customer Name
Group Name:Customer Details--Level Name:Source Of Credit--Description:Source Of Credit
Group Name:Customer Details--Level Name:Industry--Description:Industry
Group Name:Customer Details--Level Name:Cheque_type--Description:Cheque Type
Group Name:Customer Details--Level Name:Account Number--Description:Account Number
Group Name:Customer Details--Level Name:NTB Customer ID--Description:New to Bank (NTB) Customer ID
Group Name:Customer Details--Level Name:Customer Classification--Description:Customer Classification
Group Name:Customer Details--Level Name:Customer PAN--Description:Customer PAN
Group Name:Customer Details--Level Name:Customer CIN--Description:Customer CIN
Group Name:Customer Details--Level Name:NTB Case NO--Description:New to Bank (NTB) Case Number
Group Name:Customer Details--Level Name:NTB Flag--Description:New to Bank (NTB) Flag
Group Name:RM--Level Name:RM Name--Description:Relationship Manager - RM Name
Group Name:RM--Level Name:Regional Head--Description:Regional Head
Group Name:RM--Level Name:SRM--Description:Senior Relationship Manager - SRM
Group Name:RM--Level Name:Business Head--Description:Business Head
Group Name:RM--Level Name:Area Head--Description:Area Head
Group Name:Collaterals--Level Name:Security Category--Description:Security Category of the collateral
Group Name:Collaterals--Level Name:Security Sub Category--Description:Security Sub Category of the collateral
Group Name:Collaterals--Level Name:Security Type--Description:Security Type of the collateral
"Group Name:Financial IBBI--Level Name:Announcement Type--Description:Insolvency and Bankruptcy Board of India (IBBI) - Announcement Type like Announcement of Corporate Insolvency Resolution Process, Announcement of Voluntary Liquidation Process etc."
Group Name:Financial IBBI--Level Name:Applicant Name--Description:Insolvency and Bankruptcy Board of India (IBBI) -Applicant Name
Group Name:External Funding--Level Name:External Fund ISIN--Description:External Fund Security ISIN Number  (as per International Securities Identification Numbering System)
"Group Name:External Funding--Level Name:Rating--Description:Rating of the External Fund Security ISIN Number E.g. Crisil A+, Care A- etc."
Group Name:External Funding--Level Name:External Fund Instrument Type--Description:External Fund Instrument Type like Commerical Paper etc.
Group Name:Income Type--Level Name:Income Type--Description:Income Type
Group Name:Industry Details--Level Name:Sector Name--Description:Sector Name - Name of the Sector the customer belongs to like Energy, Consumer Discretionary, Financials, Healthcare etc.
Group Name:Industry Details--Level Name:Industry Group Name--Description:Industry Group Name - Name of the industry group of the customer / borrower. For e.g. Automobiles and components, consumer durable, insurance
Group Name:Industry Details--Level Name:Industry Name--Description:Industry Name of the customer/borrower
Group Name:Industry Details--Level Name:Sub Industry Name--Description:Sub Industry Name of the customer / borrower
Group Name:Litigation Details--Level Name:Standard Stage Of Case--Description:Standard Stage Of Case
Group Name:Time Levels--Level Name:Fiscal Month No--Description:Fiscal Month No
Group Name:Time Levels--Level Name:Day of Month--Description:Day of Month
Group Name:Account Opening Date--Level Name:Datesk--Description:Account Opening Date in YYYYMMDD format
Group Name:Time--Level Name:Year--Description:Generic Time Dimension - Year
Group Name:Time--Level Name:Quarter--Description:Generic Time Dimension - Quarter
Group Name:Time--Level Name:DateSk--Description:Generic Time Dimension - Date in YYYYMMDD format
Group Name:Time--Level Name:Day--Description:Generic Time Dimension - Day
Group Name:Time--Level Name:Month--Description:Generic Time Dimension - Month
Group Name:limits--Level Name:Exposure Type--Description:Credit Limits of Customer / Borrower or Account - Exposure Type
Group Name:limits--Level Name:Facility Activation Date--Description:Credit Limits of Customer / Borrower or Account - Facility Activation Date
Group Name:limits--Level Name:Facility Maturity Date--Description:Credit Limits of Customer / Borrower or Account -Facility Maturity Date
Group Name:limits--Level Name:Std Product Code--Description:Credit Limits of Customer / Borrower or Account - Standard Product Code
"Group Name:EPF--Level Name:EPF Defaulter Flag--Description:Employee Provident Fund - EPF Defaulter Flag(Yes, No)"
"Group Name:EPF--Level Name:EPF Delay Flag--Description:Employee Provident Fund - EPF Delay Flag (Yes, No)"
Group Name:EPF--Level Name:EPF Delay Month flag--Description:Employee Provident Fund - EPF Delay Month flag (Yes, No)
Group Name:Account Details--Level Name:Account And Beneficiary Code--Description:Account And Beneficiary Code for a transaction
"""
        self.measures = """Group Name:TDS Data--Level Name:TDS Due Day Count--Description:TDS Due Day Count
Group Name:TDS Data--Level Name:days--Description:days
Group Name:Benchmark Index--Level Name:NIFTY 500--Description:NIFTY 500
Group Name:Benchmark Index--Level Name:SENSEX 50--Description:SENSEX 50
Group Name:Benchmark Index--Level Name:Index Close Price--Description:Index Close Price for a given Index Name
Group Name:Financial Contract Events--Level Name:Event Type I--Description:Event Type I
Group Name:Financial Contract Events--Level Name:Event Type D--Description:Event Type D
Group Name:Financial Contract Events--Level Name:Invocation Of Bank Guarantee--Description:Invocation Of Bank Guarantee
Group Name:Financial Contract Events--Level Name:Devolvement of letter of credit--Description:Devolvement of letter of credit
Group Name:Financial Data--Level Name:Working Capital--Description:Working Capital
Group Name:Financial Data--Level Name:Profit Before Tax--Description:Profit Before Tax
Group Name:Financial Data--Level Name:Cash Equivalent LY--Description:Cash Equivalent LY
Group Name:Financial Data--Level Name:Days Debators EST--Description:Days Debators EST
Group Name:Financial Data--Level Name:Current Portion Of Long Term Debit--Description:Current Portion Of Long Term Debit
Group Name:Financial Data--Level Name:Short Term Borrowing LY--Description:Short Term Borrowing LY
Group Name:Financial Data--Level Name:Cash Accrual from Operation--Description:Cash Accrual from Operation
Group Name:Financial Data--Level Name:Days Debators--Description:Days Debators
Group Name:Financial Data--Level Name:Net Cash Accruals--Description:Net Cash Accruals
Group Name:Financial Data--Level Name:Total Current Liabilities--Description:Total Current Liabilities
Group Name:Financial Data--Level Name:Finance Cost--Description:Finance Cost
Group Name:Financial Data--Level Name:Financial Debtors Days--Description:Financial Debtors Days
Group Name:Financial Data--Level Name:Total Revenue From Operations--Description:Total Revenue From Operations
Group Name:Financial Data--Level Name:EBITDA Percentage--Description:EBITDA Percentage
Group Name:Financial Data--Level Name:Total revenue From Operations EST--Description:Total revenue From Operations EST
Group Name:Financial Data--Level Name:Other Current Assets Latest Year--Description:Other Current Assets Latest Year
Group Name:Financial Data--Level Name:Total Revenue--Description:Total Revenue
Group Name:Financial Data--Level Name:Total Debit--Description:Total Debit
Group Name:Financial Data--Level Name:Change in Working Capital--Description:Change in Working Capital
Group Name:Financial Data--Level Name:Cash Equivalent PY--Description:Cash Equivalent PY
Group Name:Financial Data--Level Name:Increase Cash Equivalent--Description:Increase Cash Equivalent
Group Name:Financial Data--Level Name:DSCR--Description:DSCR
Group Name:Financial Data--Level Name:Operating Cash Flow To Current Liabilities--Description:Operating Cash Flow To Current Liabilities
Group Name:Financial Data--Level Name:Increase in holding levels--Description:Increase in holding levels
Group Name:Financial Data--Level Name:Total Revenue From Operations PY--Description:Total Revenue From Operations PY
Group Name:Financial Data--Level Name:Change in other Current Assest--Description:Change in other Current Assest
Group Name:Financial Data--Level Name:Days Debators Deviation--Description:Days Debators Deviation
Group Name:Financial Data--Level Name:Working Capital PY--Description:Working Capital PY
Group Name:Financial Data--Level Name:Increase Short Term Borrowing--Description:Increase Short Term Borrowing
Group Name:Financial Data--Level Name:Operating Cash Flow To Total serviceable Debt--Description:Operating Cash Flow To Total serviceable Debt
Group Name:Financial Data--Level Name:Other Current Asset in Prev Year--Description:Other Current Asset in Prev Year
Group Name:Financial Data--Level Name:Cash Flow--Description:Cash Flow
Group Name:Financial Data--Level Name:Increase In Total Debt--Description:Increase In Total Debt
Group Name:Financial Data--Level Name:Total Revenue Operation dev--Description:Total Revenue Operation dev
Group Name:Financial Data--Level Name:Total Debt PY--Description:Total Debt PY
Group Name:Financial Data--Level Name:Short Term Borrowing PY--Description:Short Term Borrowing PY
Group Name:Financial Data--Level Name:Increase in Borrowing despite Huge Cash Equi--Description:Increase in Borrowing despite Huge Cash Equi
Group Name:Financial Data--Level Name:Shortfall in Net Sale--Description:Shortfall in Net Sale
Group Name:NBFC Financial Ratios--Level Name:NBFC Total Debit ADJ Net Worth--Description:NBFC Total Debit ADJ Net Worth
Group Name:NBFC Financial Ratios--Level Name:NBFC Prev NIM Funds Deployed--Description:NBFC Prev NIM Funds Deployed
Group Name:NBFC Financial Ratios--Level Name:NBFC NIM Average Funds Deployed--Description:NBFC NIM Average Funds Deployed
Group Name:NBFC Financial Ratios--Level Name:NBFC Gross NPA Trend--Description:NBFC Gross NPA Trend
Group Name:NBFC Financial Ratios--Level Name:NBFC Gross NPA--Description:NBFC Gross NPA
Group Name:NBFC Financial Ratios--Level Name:NBFC Prev Gross NPA--Description:NBFC Prev Gross NPA
Group Name:NBFC Financial Ratios--Level Name:NBFC NIM Average Funds Deployed Trend--Description:NBFC NIM Average Funds Deployed Trend
Group Name:NBFC Financial Ratios--Level Name:NBFC Gross NPA Trend EOP--Description:NBFC Gross NPA Trend EOP
Group Name:NBFC Financial Ratios--Level Name:NBFC NIM Average Funds Deployed Trend EOP--Description:NBFC NIM Average Funds Deployed Trend EOP
Group Name:EPF Mesures--Level Name:Provident Fund Due Date Count--Description:Provident Fund Due Date Count
Group Name:EPF Mesures--Level Name:Number Of Days Delay--Description:Number Of Days Delay
Group Name:EPF Mesures--Level Name:Employee Count Previous Month--Description:Employee Count Previous Month
Group Name:EPF Mesures--Level Name:EPF Due Days--Description:Employee Provident Fund (EPF) Due Days
Group Name:EPF Mesures--Level Name:Employee Count Percentage Change MOM--Description:Employee Count Percentage Change MOM
Group Name:EPF Mesures--Level Name:Previous Employee Amount--Description:Previous Employee Amount
Group Name:EPF Mesures--Level Name:EPF Amount--Description:Employee Provident Fund (EPF) Amount
Group Name:EPF Mesures--Level Name:Employee Count--Description:Employee Count
Group Name:EPF Mesures--Level Name:Number Of Times Days Delay--Description:Number Of Times Days Delay
Group Name:EPF Mesures--Level Name:Employee Amount Percentage Change MOM--Description:Employee Amount Percentage Change MOM
Group Name:EPF Mesures--Level Name:Employee Percentage Amount EOP--Description:Employee Percentage Amount EOP
Group Name:EPF Mesures--Level Name:Employee Percentage Count EOP--Description:Employee Percentage Count EOP
Group Name:EPF Mesures--Level Name:Employee Amount 6M  % Change--Description:Employee Amount 6M  % Change
Group Name:EPF Mesures--Level Name:Employee Count 6M % Change--Description:Employee Count 6M % Change
Group Name:EPF Mesures--Level Name:EPF Defaulter EOP--Description:Employee Provident Fund (EPF) Defaulter EOP
Group Name:NCLT Details--Level Name:Count of cases in NCLT--Description:Count of cases in NCLT
Group Name:NCLT Details--Level Name:Diposed Count--Description:Diposed Count
Group Name:NCLT Details--Level Name:Pending  Count--Description:Pending  Count
Group Name:NCLT Details--Level Name:Total Count Of NCLT Status--Description:Total Count Of NCLT Status
Group Name:Auditor Details--Level Name:Auditor Membership Number--Description:Auditor Membership Number
Group Name:Auditor Details--Level Name:Delay In Days--Description:Maximum Delay In Days
Group Name:Auditor Details--Level Name:Change in Auditor member Number--Description:Change in Auditor member Number
Group Name:Auditor Details--Level Name:Prev Auditor Membership Number--Description:Previous Auditor Membership Number
Group Name:Auditor Details--Level Name:Frequent changes in the statutory auditors of the--Description:Frequent changes in the statutory auditors of the customer/borrower
Group Name:Auditor Details--Level Name:Max Frequent changes in the statutory auditors of--Description:Maximum Frequent changes in the statutory auditors 
Group Name:Auditor Details--Level Name:Non uploading of the Audited Financial Statements--Description:Non uploading of the Audited Financial Statements
Group Name:Suit Filed Details--Level Name:First OS Amount--Description:First OS Amount
Group Name:Suit Filed Details--Level Name:Last OS Amount--Description:Last OS Amount
Group Name:Suit Filed Details--Level Name:Count Of CIN--Description:Count Of CIN
Group Name:Suit Filed Details--Level Name:Suit Filed--Description:Suit Filed
Group Name:Financial Ratios--Level Name:Trade Receivable To Sales--Description:Trade Receivable To Sales
Group Name:Financial Ratios--Level Name:Return On Equity--Description:Return On Equity
Group Name:Financial Ratios--Level Name:Cashflow Margin--Description:Cashflow Margin
Group Name:Financial Ratios--Level Name:TOL TNW--Description:TOL TNW
Group Name:Financial Ratios--Level Name:Total Debit Net Cash Accural--Description:Total Debit Net Cash Accural
Group Name:Financial Ratios--Level Name:NET FIXED ASSETS LY--Description:NET FIXED ASSETS LY
Group Name:Financial Ratios--Level Name:Interest Coverage--Description:Interest Coverage
Group Name:Financial Ratios--Level Name:Receivables Turnover--Description:Receivables Turnover
Group Name:Financial Ratios--Level Name:Cash Ratio--Description:Cash Ratio
Group Name:Financial Ratios--Level Name:Days Payable Outstanding--Description:Days Payable Outstanding
Group Name:Financial Ratios--Level Name:Total Asset Growth--Description:Total Asset Growth
Group Name:Financial Ratios--Level Name:Return On Asset--Description:Return On Asset
Group Name:Financial Ratios--Level Name:Current Ratio Peer--Description:Current Ratio Peer
Group Name:Financial Ratios--Level Name:Payable Turnover--Description:Payable Turnover
Group Name:Financial Ratios--Level Name:Return On Capital Emplyed--Description:Return On Capital Emplyed
Group Name:Financial Ratios--Level Name:Current Liability Coverage--Description:Current Liability Coverage
Group Name:Financial Ratios--Level Name:EBITDA Coverage--Description:EBITDA Coverage
Group Name:Financial Ratios--Level Name:PAT Margin Percent--Description:PAT Margin Percent
Group Name:Financial Ratios--Level Name:Day Sales Outstanding--Description:Day Sales Outstanding
Group Name:Financial Ratios--Level Name:Equity Multiplier--Description:Equity Multiplier
Group Name:Financial Ratios--Level Name:Total Asset Turnover--Description:Total Asset Turnover
Group Name:Financial Ratios--Level Name:Total debt or Equity EST--Description:Total debt or Equity EST
Group Name:Financial Ratios--Level Name:EBITDA Margin Percent--Description:EBITDA Margin Percent
Group Name:Financial Ratios--Level Name:Debit To EBITDA--Description:Debit To EBITDA
Group Name:Financial Ratios--Level Name:Cashflow Coverage--Description:Cashflow Coverage
Group Name:Financial Ratios--Level Name:EBITDA MARGINS PERCENT--Description:EBITDA MARGINS PERCENT
Group Name:Financial Ratios--Level Name:Cash Conversion Cycle--Description:Cash Conversion Cycle
Group Name:Financial Ratios--Level Name:Total Debt Or Equity--Description:Total Debt Or Equity
Group Name:Financial Ratios--Level Name:EBITDA Margin Percent EST--Description:EBITDA Margin Percent EST
Group Name:Financial Ratios--Level Name:Net Profit Margin EST--Description:Net Profit Margin EST
Group Name:Financial Ratios--Level Name:Net Worth Growth--Description:Net Worth Growth
Group Name:Financial Ratios--Level Name:Total Liability Growth--Description:Total Liability Growth
Group Name:Financial Ratios--Level Name:TOT REV FRM OPER LY--Description:TOT REV FRM OPER LY
Group Name:Financial Ratios--Level Name:Working Capital Turnover--Description:Working Capital Turnover
Group Name:Financial Ratios--Level Name:Quick Ratio--Description:Quick Ratio
Group Name:Financial Ratios--Level Name:Total Revenue From Operation--Description:Total Revenue From Operation
Group Name:Financial Ratios--Level Name:Revenue Growth--Description:Revenue Growth
Group Name:Financial Ratios--Level Name:Inventory CY--Description:Inventory CY
Group Name:Financial Ratios--Level Name:Debit Capital--Description:Debit Capital
Group Name:Financial Ratios--Level Name:TOL TNW EST--Description:TOL TNW EST
Group Name:Financial Ratios--Level Name:Gross Profit Margin--Description:Gross Profit Margin
Group Name:Financial Ratios--Level Name:Return On Equity EST--Description:Return On Equity EST
Group Name:Financial Ratios--Level Name:Inventory Days EST--Description:Inventory Days EST
Group Name:Financial Ratios--Level Name:Trade Receivables--Description:Trade Receivables
Group Name:Financial Ratios--Level Name:Inventory Turnover--Description:Inventory Turnover
Group Name:Financial Ratios--Level Name:Days Working Capital--Description:Days Working Capital
Group Name:Financial Ratios--Level Name:Long Term Debit Coverage--Description:Long Term Debit Coverage
Group Name:Financial Ratios--Level Name:Long Term Debit Equity--Description:Long Term Debit Equity
Group Name:Financial Ratios--Level Name:Net Profit Margin--Description:Net Profit Margin
Group Name:Financial Ratios--Level Name:Current Ratio--Description:Current Ratio
Group Name:Financial Ratios--Level Name:Gross Curr Assets Days Sales--Description:Gross Curr Assets Days Sales
Group Name:Financial Ratios--Level Name:Total By ANW--Description:Total By ANW
Group Name:Financial Ratios--Level Name:Inventory Days--Description:Inventory Days
Group Name:Financial Ratios--Level Name:TOL TNW EST EOP--Description:TOL TNW EST EOP
Group Name:Financial Ratios--Level Name:Increase NET FIXED ASSETS--Description:Increase NET FIXED ASSETS
Group Name:Financial Ratios--Level Name:Current Ratio EOP--Description:Current Ratio EOP
Group Name:Financial Ratios--Level Name:Total Revenue From Operation PY--Description:Total Revenue From Operation PY
Group Name:Financial Ratios--Level Name:Days Inventories--Description:Days Inventories
Group Name:Financial Ratios--Level Name:Total Debt or Equity Deviation--Description:Total Debt or Equity Deviation
Group Name:Financial Ratios--Level Name:LY EBITDA MARGINS PERCENT--Description:LY EBITDA MARGINS PERCENT
Group Name:Financial Ratios--Level Name:Total By ANW EOP--Description:Total By ANW EOP
Group Name:Financial Ratios--Level Name:TOL TNW EOP--Description:TOL TNW EOP
Group Name:Financial Ratios--Level Name:Interest Coverage EOP--Description:Interest Coverage EOP
Group Name:Financial Ratios--Level Name:Inventories Cost sales Downward--Description:Inventories Cost sales Downward
Group Name:Financial Ratios--Level Name:Return On Equity EST EOP--Description:Return On Equity EST EOP
Group Name:Financial Ratios--Level Name:Receivables Growth--Description:Receivables Growth
Group Name:Financial Ratios--Level Name:NET FIXED ASSETS PY--Description:NET FIXED ASSETS PY
Group Name:Financial Ratios--Level Name:RONW Est--Description:RONW Est
Group Name:Financial Ratios--Level Name:EBITDA Margin Percent EST EOP--Description:EBITDA Margin Percent EST EOP
Group Name:Financial Ratios--Level Name:Days Inventories EOP--Description:Days Inventories EOP
Group Name:Financial Ratios--Level Name:Total Debt Or Equity EOP--Description:Total Debt Or Equity EOP
Group Name:Financial Ratios--Level Name:Net Profit Margin Deviation--Description:Net Profit Margin Deviation
Group Name:Financial Ratios--Level Name:Inventory Growth--Description:Inventory Growth
Group Name:Financial Ratios--Level Name:Trade Receivables PY--Description:Trade Receivables PY
Group Name:Financial Ratios--Level Name:Significant Movement In Recivables--Description:Significant Movement In Recivables
Group Name:Financial Ratios--Level Name:Increase TOT REV FRM OPER--Description:Increase TOT REV FRM OPER
Group Name:Financial Ratios--Level Name:Revenue Turnover Growth--Description:Revenue Turnover Growth
Group Name:Financial Ratios--Level Name:Quick Ratio EOP--Description:Quick Ratio EOP
Group Name:Financial Ratios--Level Name:ADJ_TOL_TNW--Description:ADJ_TOL_TNW
Group Name:Financial Ratios--Level Name:Return On Equity EOP--Description:Return On Equity EOP
Group Name:Financial Ratios--Level Name:EBITDA Margin sub--Description:EBITDA Margin sub
Group Name:Financial Ratios--Level Name:Prev Trade Receivable To Sales--Description:Prev Trade Receivable To Sales
Group Name:Financial Ratios--Level Name:Inventory Days EOP--Description:Inventory Days EOP
Group Name:Financial Ratios--Level Name:significant Inventory  Growth--Description:significant Inventory  Growth
Group Name:Financial Ratios--Level Name:TOT REV FRM OPER PY--Description:TOT REV FRM OPER PY
Group Name:Financial Ratios--Level Name:Inventory Days EST EOP--Description:Inventory Days EST EOP
Group Name:Financial Ratios--Level Name:Total Asset Turnover EOP--Description:Total Asset Turnover EOP
Group Name:Financial Ratios--Level Name:EBITDA Percentage Change--Description:EBITDA Percentage Change
Group Name:Financial Ratios--Level Name:Net Profit Margin EOP--Description:Net Profit Margin EOP
Group Name:Financial Ratios--Level Name:Net Profit Margin EST EOP--Description:Net Profit Margin EST EOP
Group Name:Financial Ratios--Level Name:Inc in NetFA without Inc in Turnover--Description:Inc in NetFA without Inc in Turnover
Group Name:Financial Ratios--Level Name:Return On Capital Emplyed EOP--Description:Return On Capital Emplyed EOP
Group Name:Financial Ratios--Level Name:Inventory PY--Description:Inventory PY
Group Name:Financial Ratios--Level Name:Latest Day Inventory--Description:Latest Day Inventory
Group Name:Financial Ratios--Level Name:Net Cash Accural EOP--Description:Net Cash Accural EOP
Group Name:Financial Ratios--Level Name:Significant Growth on Revenue--Description:Significant Growth on Revenue
Group Name:Financial Ratios--Level Name:EBITDA Margin Percent EOP--Description:EBITDA Margin Percent EOP
Group Name:Risk Mitigants--Level Name:Product Limit Sum--Description:Product Limit Sum
Group Name:Risk Mitigants--Level Name:Non Funded Limit Sum--Description:Non Funded Limit Sum
Group Name:Risk Mitigants--Level Name:Limit Expiry Count--Description:Limit Expiry Count
Group Name:Risk Mitigants--Level Name:Non Funded Limit--Description:Non Funded Limit
Group Name:Risk Mitigants--Level Name:Funded Limit--Description:Funded Limit
Group Name:Risk Mitigants--Level Name:Base WC Turnover--Description:Base WC Turnover
Group Name:Risk Mitigants--Level Name:Collateral Market Value--Description:Collateral Market Value
Group Name:Risk Mitigants--Level Name:Collateral Value--Description:Collateral Value
Group Name:Risk Mitigants--Level Name:Sanction Limit Apr FY--Description:Sanction Limit Apr FY
Group Name:Risk Mitigants--Level Name:Product Limit--Description:Product Limit
Group Name:Risk Mitigants--Level Name:Sanction Limit Apr--Description:Sanction Limit Apr
Group Name:Risk Mitigants--Level Name:Customer Limit--Description:Customer Limit
Group Name:Fact Triggers--Level Name:Cash Withdrawal Count--Description:Cash Withdrawal Count
Group Name:Fact Triggers--Level Name:No Credit Trasactions in the Account--Description:No Credit Trasactions in the Account
Group Name:Fact Triggers--Level Name:Same Collateral to Many Lendars--Description:Same Collateral to Many Lendars
Group Name:Fact Triggers--Level Name:Large Cash Withdrawal--Description:Large Cash Withdrawal
Group Name:Fact Triggers--Level Name:Resign Of Key Person Management--Description:Resign Of Key Person Management
Group Name:Fact Triggers--Level Name:Non renewal of facilities--Description:Non renewal of facilities
Group Name:Fact Triggers--Level Name:CW Large Cash--Description:Cash Withdrawal (CW) Large Cash
Group Name:Customer Questions--Level Name:Invoices devoid of TAN and other details--Description:Invoices devoid of TAN and other details
Group Name:Customer Questions--Level Name:Unfavourable trends in borrower value chain--Description:Unfavourable trends in borrower value chain
Group Name:Customer Questions--Level Name:Crystallization of Export Bills--Description:Crystallization of Export Bills
Group Name:Customer Questions--Level Name:Serious qualification in the Bank Statutory Audito--Description:Serious qualification in the Bank Statutory Audito
Group Name:Customer Questions--Level Name:Critical issues highlighted in Statutory Audit Rep--Description:Critical issues highlighted in Statutory Audit Rep
Group Name:Customer Questions--Level Name:Significant inconsistencies within the annual repo--Description:Significant inconsistencies within the annual repo
Group Name:Customer Questions--Level Name:Non submission of stock and book debts statements--Description:Non submission of stock and book debts statements
Group Name:Customer Questions--Level Name:Dispute on title of the collateral securities--Description:Dispute on title of the collateral securities
Group Name:Customer Questions--Level Name:Large number of transactions with inter-connected--Description:Large number of transactions with inter-connected
Group Name:Customer Questions--Level Name:Onerous clauses in issue of BC/LG/SBLC--Description:Onerous clauses in issue of BC/LG/SBLC
Group Name:Customer Questions--Level Name:NMaxon cooperative attitude towards stock audit--Description:NMaxon cooperative attitude towards stock audit
Group Name:Customer Questions--Level Name:Danger of technology obsolescence or introduction--Description:Danger of technology obsolescence or introduction
Group Name:Customer Questions--Level Name:Raid by Income tax sales tax--Description:Raid by Income tax sales tax
Group Name:Customer Questions--Level Name:Costing of the project which is in wide variance--Description:Costing of the project which is in wide variance
Group Name:Customer Questions--Level Name:Funds coming from other--Description:Funds coming from other
Group Name:Customer Questions--Level Name:In merchanting trade import leg not revealed--Description:In merchanting trade import leg not revealed
Group Name:Customer Questions--Level Name:Poor disclosure of materially--Description:Poor disclosure of materially
Group Name:Customer Questions--Level Name:Request received from the borrower to post pone th--Description:Request received from the borrower to post pone th
Group Name:Customer Questions--Level Name:Claims not acknowledged as debts is high--Description:Claims not acknowledged as debts is high
Group Name:Customer Questions--Level Name:Frequent utility disruption--Description:Frequent utility disruption
Group Name:Customer Questions--Level Name:Non submission of original bills--Description:Non submission of original bills
Group Name:Customer Questions--Level Name:Frequent request for general purpose--Description:Frequent request for general purpose
Group Name:Customer Questions--Level Name:Material discrepancies in the annual--Description:Material discrepancies in the annual
Group Name:Customer Questions--Level Name:Substantial increase in unbilled revenue year--Description:Substantial increase in unbilled revenue year
Group Name:Customer Questions--Level Name:Pledging/ selling of promoters' shares in the borr--Description:Pledging/ selling of promoters' shares in the borr
Group Name:Customer Questions--Level Name:Frequent change in scope of the project--Description:Frequent change in scope of the project
Group Name:Customer Questions--Level Name:Non compliance with sanction terms--Description:Non compliance with sanction terms
Group Name:Customer Questions--Level Name:Presence of other un-related bank?s name board--Description:Presence of other un-related bank?s name board
Group Name:Customer Questions--Level Name:Financing the unit far away from the branch--Description:Financing the unit far away from the branch
Group Name:Customer Questions--Level Name:Concealment of certain vital documents--Description:Concealment of certain vital documents
Group Name:Customer Questions--Level Name:Request received from the borrower--Description:Request received from the borrower
Group Name:Customer Questions--Level Name:Discrepancies in audited financial statements subm--Description:Discrepancies in audited financial statements subm
Group Name:Customer Questions--Level Name:Instance of loss of major borrower or customer--Description:Instance of loss of major borrower or customer
Group Name:Customer Questions--Level Name:Regulatory changes affecting the industry--Description:Regulatory changes affecting the industry
Group Name:Customer Questions--Level Name:Negative News other than specific for Borrower/Ind--Description:Negative News other than specific for Borrower/Ind
Group Name:Customer Questions--Level Name:High rejection of goods--Description:High rejection of goods
Group Name:Customer Questions--Level Name:Indication of fraud--Description:Indication of fraud
Group Name:Customer Questions--Level Name:Movement of an account from one bank--Description:Movement of an account from one bank
Group Name:Customer Questions--Level Name:Default in payment--Description:Default in payment
Group Name:Share Price BSE--Level Name:Share Price MoM--Description:Share Price MoM
Group Name:Share Price BSE--Level Name:Share Price QoQ--Description:Share Price QoQ
Group Name:Share Price BSE--Level Name:Share Price Deviation MoM--Description:Share Price Deviation MoM
Group Name:Share Price BSE--Level Name:Share Price Deviation QoQ--Description:Share Price Deviation QoQ
Group Name:Share Price BSE--Level Name:Sum Of Close Price--Description:Sum Of Close Price
Group Name:Share Price BSE--Level Name:Close Price--Description:Close Price
Group Name:Share Price BSE--Level Name:Previous Close Price--Description:Previous Close Price
Group Name:Share Price BSE--Level Name:Latest Quarter Share Price--Description:Latest Quarter Share Price
Group Name:Share Price BSE--Level Name:Previous Quarter Share Price--Description:Previous Quarter Share Price
Group Name:Share Price BSE--Level Name:Decline in Share Price In Last Quarter--Description:Decline in Share Price In Last Quarter
Group Name:Financial Statutary Dues--Level Name:Petitioners Name Count--Description:Petitioners Name Count
Group Name:Financial Limits--Level Name:Facility start date count--Description:Facility start date count
Group Name:Financial Limits--Level Name:Outstanding Amount--Description:Outstanding Amount
Group Name:Financial IBBI Details--Level Name:Applicant Count--Description:Insolvency and Bankruptcy Board of India (IBBI) - Applicant Count
Group Name:Fuzzy--Level Name:Local trade/related party transactions--Description:Local trade/related party transactions
Group Name:Fuzzy--Level Name:LCs issued for local trade/related party transacti--Description:Letter of Credit (LCs) issued for local trade/related party transacti
Group Name:Financial Charges--Level Name:Satisfied Charge Count--Description:Satisfied Charge Count
Group Name:Financial Charges--Level Name:Open Charge Count--Description:Open Charge Count
Group Name:Financial Charges--Level Name:Charge Amount--Description:Charge Amount
Group Name:Financial Charges--Level Name:Latest Charge Amount--Description:Latest Charge Amount
Group Name:Financial Charges--Level Name:Latest Charge Amount PY--Description:Latest Charge Amount Previous Year (PY)
Group Name:Financial Charges--Level Name:Change in Total Charge Amount--Description:Change in Total Charge Amount
Group Name:Financial Charges--Level Name:Liabilities Appearing in ROC Search Report--Description:Liabilities Appearing in Registrar of Companies (ROC) Search Report
Group Name:Financial Charges--Level Name:Max Liabilities Appearing in ROC search--Description:Max Liabilities Appearing in Registrar of Companies (ROC) search
Group Name:Bulk Deal Trade--Level Name:Traded Quantity--Description:Bulk Deal - Traded Quantity of Security
Group Name:Bulk Deal Trade--Level Name:Trade Price--Description:Bulk Deal - Trade Price in which the bulk deal has happened
Group Name:Bulk Deal Trade--Level Name:Count Of TDS--Description:TDS Details - Count Of Tax Deducted at Source
Group Name:Financial Collateral--Level Name:PY Financial Collateral Value--Description:PY Financial Collateral Value
Group Name:Financial Collateral--Level Name:Financial Collateral--Description:Financial Collateral
Group Name:Financial Collateral--Level Name:Pending Perfection Security EOP--Description:Pending Perfection Security EOP
Group Name:Financial Collateral--Level Name:Bureau DPD EOP--Description:Bureau Days Past Due (DPD) EOP
Group Name:Financial Yield Movement--Level Name:Bond Yeild 6M % Change--Description:Bond Yeild 6M % Change
Group Name:Financial Yield Movement--Level Name:Previous Bound Yeild MOM--Description:Previous Bound Yeild MOM
Group Name:Financial Yield Movement--Level Name:Bond Yeild In Last 6 Month--Description:Bond Yeild In Last 6 Month
Group Name:Financial Yield Movement--Level Name:Previous Bond Yeild 6M--Description:Previous Bond Yeild 6M
Group Name:Financial Yield Movement--Level Name:Bond Yeild Price--Description:Bond Yeild Price
Group Name:Financial Yield Movement--Level Name:Bond Yeild MOM % Change--Description:Bond Yeild MOM % Change
Group Name:Financial Yield Movement--Level Name:Bond Yeild MOM % EOP--Description:Bond Yeild MOM % EOP
Group Name:Financial Yield Movement--Level Name:Bond Yeild 6M % EOP--Description:Bond Yeild 6M % EOP
Group Name:Google News--Level Name:Withdrawal by Project--Description:Withdrawal by Project
Group Name:Google News--Level Name:Raid by Income Tax--Description:Raid by Income Tax
Group Name:Google News--Level Name:Negative Sentiments--Description:Negative Sentiments
Group Name:Google News--Level Name:High Sentiment Count--Description:High Sentiment Count
Group Name:Google News--Level Name:Low Sentiment Count--Description:Low Sentiment Count
Group Name:Google News--Level Name:Google News Score--Description:Google News Score
Group Name:Google News--Level Name:Medium Sentiment Count--Description:Medium Sentiment Count
Group Name:Google News--Level Name:Instance Of Loss Borrower--Description:Instance Of Loss Borrower
Group Name:Google News--Level Name:Regulatory_industry--Description:Regulatory_industry
Group Name:Google News--Level Name:Negative News EOP--Description:Negative News EOP
Group Name:Google News--Level Name:Regulatory_industry EOP--Description:Regulatory_industry EOP
Group Name:Google News--Level Name:Instance Of Loss Borrower EOP--Description:Instance Of Loss Borrower EOP
Group Name:External Funding Details--Level Name:CPCD Yeild--Description:Commercial Paper Certificate of Deposits (CPCD) Yeild
Group Name:External Funding Details--Level Name:CPCD Price--Description:Commercial Paper Certificate of Deposits (CPCD) Price
Group Name:External Funding Details--Level Name:Previous CPCD Yeild MoM--Description:Previous Commercial Paper Certificate of Deposits (CPCD) Yeild MoM
Group Name:External Funding Details--Level Name:%Change On CPCD Yeild QoQ--Description:%Change On Commercial Paper Certificate of Deposits (CPCD) Yeild QoQ
Group Name:External Funding Details--Level Name:CPCD  Face Value--Description:CPCD  Face Value
Group Name:External Funding Details--Level Name:%Change On CPCD Yeild MOM--Description:%Change On Commercial Paper Certificate of Deposits (CPCD) Yeild MOM
Group Name:External Funding Details--Level Name:Previous CPCD Yeild QoQ--Description:Previous Commercial Paper Certificate of Deposits (CPCD) Yeild QoQ
Group Name:External Funding Details--Level Name:CPCD Count In Last 6 Month--Description:Commercial Paper Certificate of Deposits (CPCD) Count In Last 6 Month
Group Name:External Funding Details--Level Name:CPCD Yeild MoM % Change  EOP--Description:Commercial Paper Certificate of Deposits (CPCD) Yeild MoM % Change  EOP
Group Name:External Funding Details--Level Name:CPCD Yeild QoQ % Change  EOP--Description:Commercial Paper Certificate of Deposits (CPCD) Yeild QoQ % Change  EOP
Group Name:Financial Project Details--Level Name:Account Label Weight--Description:Account Label Weight
Group Name:Credit Risk--Level Name:Overdue Amount--Description:Overdue Amount
Group Name:Credit Risk--Level Name:Overdue Days--Description:Overdue Days - Max
Group Name:Fund Investment Details--Level Name:MF Count Last 6 Month--Description:MF Count Last 6 Month
Group Name:Fund Investment Details--Level Name:Mutual Fund % Change MoM--Description:Mutual Fund % Change MoM
Group Name:Fund Investment Details--Level Name:Percent to NAV--Description:Percent to NAV
Group Name:Fund Investment Details--Level Name:Mutual Fund % Change QoQ--Description:Mutual Fund % Change QoQ
Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity QoQ--Description:Mutual Fund Quantity QoQ
Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity MoM--Description:Mutual Fund Quantity MoM
Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity--Description:Mutual Fund Quantity
Group Name:Fund Investment Details--Level Name:Fund Market Value--Description:Fund Market Value
Group Name:Fund Investment Details--Level Name:Mutual Fund MoM EOP--Description:Mutual Fund MoM EOP
Group Name:Fund Investment Details--Level Name:Mutual Fund QoQ EOP--Description:Mutual Fund QoQ EOP
Group Name:GST Filing Details--Level Name:GSTR1 Due Day Count--Description:GSTR1 Due Day Count
Group Name:GST Filing Details--Level Name:GSTR3b Due Day Count--Description:GSTR3b Due Day Count
Group Name:Account Conduct--Level Name:Frequent Delay in Adjustment of PCs--Description:Frequent Delay in Adjustment of PCs
Group Name:Account Conduct--Level Name:Frequent return of bills--Description:Frequent return of bills
Group Name:Account Conduct--Level Name:Delay in Interest Servicing--Description:Delay in Interest Servicing
Group Name:Account Conduct--Level Name:Number Of Consecutive Month  Decline in Debit--Description:Number Of Consecutive Month there is a decline in Debit
Group Name:Account Conduct--Level Name:Number Of Consecutive Month  Decline in Credit--Description:Number Of Consecutive Month there is a decline in Credit
Group Name:Account Conduct--Level Name:Excess in acct--Description:Excess in account
Group Name:Account Conduct--Level Name:Sanction amt avg--Description:Sanction Amount Average
Group Name:Account Conduct--Level Name:LC_BG_NO--Description:Letter of Credit (LC) / Bank Guarantee (BG) Number
Group Name:Account Conduct--Level Name:interest_overdue--Description:Interest Overdue
Group Name:Account Conduct--Level Name:Overdrawn in Accounts--Description:Overdrawn in Accounts
Group Name:Account Conduct--Level Name:Credit Summation Sum--Description:Credit Summation
Group Name:Account Conduct--Level Name:Credit Summation And Sanction Limit--Description:Credit Summation And Sanction Limit
Group Name:Account Conduct--Level Name:Not routing sales through our bank account--Description:Not routing sales through our bank account
Group Name:Account Conduct--Level Name:Excess during quarter--Description:Excess during quarter
Group Name:Account Conduct--Level Name:Excess during month--Description:Excess during month
Group Name:Account Conduct--Level Name:Continuous excess in working capital account--Description:Continuous excess in working capital account
Group Name:Account Conduct--Level Name:Continuous over dues in the term loans--Description:Continuous over dues in the term loans
Group Name:Account Conduct--Level Name:Credit Summation Sum Last 30 Days--Description:Credit Summation Sum Last 30 Days
Group Name:Financial Shareholding--Level Name:Shares Pledged %--Description:Average Shares Pledged %
Group Name:Financial Shareholding--Level Name:Corp Shareholding Quantity--Description:Corporate Shareholding Quantity
Group Name:Financial Shareholding--Level Name:Previous Promoter Shareholding %--Description:Previous Promoter Shareholding %
Group Name:Financial Shareholding--Level Name:Promoter Shareholding Percentage--Description:Promoter Shareholding Percentage
Group Name:Financial Shareholding--Level Name:Promoter Shareholding %--Description:Promoter Shareholding %
Group Name:Financial Shareholding--Level Name:Corp shareholding percentage--Description:Corp shareholding percentage
Group Name:Financial Shareholding--Level Name:Sharehold Percent Of Total Shares--Description:Sharehold Percent Of Total Shares
Group Name:Financial Shareholding--Level Name:Number of Shares Pledged--Description:Number of Shares Pledged
Group Name:Financial Shareholding--Level Name:Promoter Share Pledging--Description:Promoter Share Pledging
Group Name:Financial Shareholding--Level Name:Shares Pledged % EOP--Description:Shares Pledged %  - End of Period (EOP)
Group Name:Financial Shareholding--Level Name:Change in Shareholding of the Borrower--Description:Change in Shareholding of the Borrower
Group Name:Financial Shareholding--Level Name:Corp shareholding Percentage EOP--Description:Corp shareholding Percentage - End of Period (EOP)
Group Name:Financial Shareholding--Level Name:CY Share Hold Percent Total Share--Description:Current Year (CY) Share Hold Percent Total Share
Group Name:Financial Shareholding--Level Name:Promoter Shareholding % EOP--Description:Promoter Shareholding % - End of Period (EOP)
Group Name:Financial Shareholding--Level Name:PY Share Hold Percent Total Share--Description:Previous Year (PY) Share Hold Percent Total Share
Group Name:DRT--Level Name:DRT Disposed Count--Description:Debt Recovery Tribunal (DRT)  Disposed Count
Group Name:DRT--Level Name:DRT Pending Count--Description:Debt Recovery Tribunal (DRT)  Pending Count
Group Name:DRT--Level Name:DRT Total Status Count--Description:Debt Recovery Tribunal (DRT)  Total Status Count
Group Name:Stock Statement--Level Name:Stock Insurance--Description:Stock Insurance
Group Name:Stock Statement--Level Name:Average Of Sanction Ammount--Description:Average Of Sanction Ammount
Group Name:Stock Statement--Level Name:Sanction Amount Average--Description:Sanction Amount Average
Group Name:Stock Statement--Level Name:Credit Summation--Description:Credit Summation
Group Name:Stock Statement--Level Name:Stock Days--Description:Stock Days
Group Name:Stock Statement--Level Name:Stock Insurance Check--Description:Stock Insurance Check
Group Name:Stock Statement--Level Name:Stock Value--Description:Stock Value
Group Name:Stock Statement--Level Name:Credit Summation Last 30 Days--Description:Credit Summation Last 30 Days
Group Name:Stock Statement--Level Name:FQ Non submission of stock--Description:FQ Non submission of stock
Group Name:Stock Statement--Level Name:Non submission of delay days--Description:Non submission of delay days
Group Name:Stock Statement--Level Name:High Turnover Compare To Sales--Description:High Turnover Compare To Sales
Group Name:Stock Statement--Level Name:Credit Summation and sanction limit sum--Description:Credit Summation and sanction limit sum
Group Name:Stock Statement--Level Name:Over Insured Stock--Description:Over Insured Stock
Group Name:Financial Disbursement--Level Name:Adj PC out of local funds--Description:Adjusted PC out of local funds
Group Name:Financial Disbursement--Level Name:Bills Send for Collection PC Out--Description:Bills Send for Collection PC Out
Group Name:Negative Checklist Count--Level Name:Due Diligence--Description:Due Diligence
Group Name:Negative Checklist Count--Level Name:Credit Rate Withdraw--Description:Credit Rate Withdraw
Group Name:Negative Checklist Count--Level Name:Vanishing--Description:Vanishing
Group Name:Negative Checklist Count--Level Name:Vanishing Company--Description:Vanishing Company
Group Name:Negative Checklist Count--Level Name:SEBI Debarred--Description:SEBI Debarred
Group Name:Negative Checklist Count--Level Name:MLM Company--Description:MLM Company
Group Name:Negative Checklist Count--Level Name:EPF Under Liquidation--Description:EPF Under Liquidation
Group Name:Negative Checklist Count--Level Name:EPF Due Days--Description:EPF Due Days
Group Name:Negative Checklist Count--Level Name:Open Charge ARC--Description:Open Charge ARC
Group Name:Negative Checklist Count--Level Name:MCA Defaulter--Description:MCA Defaulter
Group Name:Negative Checklist Count--Level Name:Negative Checklist Medium Count--Description:Negative Checklist Medium Count
Group Name:Negative Checklist Count--Level Name:Company Striked Off--Description:Company Striked Off
Group Name:Negative Checklist Count--Level Name:Credit Rate Suspend--Description:Credit Rate Suspend
Group Name:Negative Checklist Count--Level Name:Negative Check--Description:Negative Check
Group Name:Negative Checklist Count--Level Name:Company Under Strike Off--Description:Company Under Strike Off
Group Name:Negative Checklist Count--Level Name:Company Under Liquidation--Description:Company Under Liquidation
Group Name:Negative Checklist Count--Level Name:Offshore Leak--Description:Offshore Leak
Group Name:Negative Checklist Count--Level Name:IBBI--Description:IBBI
Group Name:Negative Checklist Count--Level Name:TAN Transaction--Description:TAN Transaction
Group Name:Negative Checklist Count--Level Name:Negative Checklist Low Count--Description:Negative Checklist Low Count
Group Name:Negative Checklist Count--Level Name:Open Charge SASF--Description:Open Charge SASF
Group Name:Negative Checklist Count--Level Name:Dormant--Description:Dormant
Group Name:Negative Checklist Count--Level Name:Shell Company--Description:Shell Company
Group Name:Negative Checklist Count--Level Name:Disqualified Director--Description:Disqualified Director
Group Name:Negative Checklist Count--Level Name:IEC Black List--Description:IEC Black List
Group Name:Negative Checklist Count--Level Name:EPF Register With BIFR--Description:EPF Register With BIFR
Group Name:Negative Checklist Count--Level Name:BIFR--Description:BIFR
Group Name:Negative Checklist Count--Level Name:Negative Checklist High Count--Description:Negative Checklist High Count
Group Name:Negative Checklist Count--Level Name:GST Transaction--Description:GST Transaction
Group Name:Negative Checklist Count--Level Name:Corp GST Transaction--Description:Corp GST Transaction
Group Name:Negative Checklist Count--Level Name:Hawala--Description:Hawala
Group Name:Negative Checklist Count--Level Name:Company Strike Off Section 248--Description:Company Strike Off Section 248
Group Name:Negative Checklist Count--Level Name:EPF Transaction--Description:EPF Transaction
Group Name:Company Scorecard Details--Level Name:Profit Score % Change--Description:Profit Score % Change
Group Name:Company Scorecard Details--Level Name:Previous Efficient Score--Description:Previous Efficient Score
Group Name:Company Scorecard Details--Level Name:Profit Score--Description:Profit Score
Group Name:Company Scorecard Details--Level Name:Overall Score--Description:Overall Score
Group Name:Company Scorecard Details--Level Name:Previous Coverage Score--Description:Previous Coverage Score
Group Name:Company Scorecard Details--Level Name:Coverage Score % Change--Description:Coverage Score % Change
Group Name:Company Scorecard Details--Level Name:Efficient Score--Description:Efficient Score
Group Name:Company Scorecard Details--Level Name:Previous Profit Score--Description:Previous Profit Score
Group Name:Company Scorecard Details--Level Name:Coverage Score--Description:Coverage Score
Group Name:Company Scorecard Details--Level Name:Previous Overall Score--Description:Previous Overall Score
Group Name:Company Scorecard Details--Level Name:Efficient Score % Change--Description:Efficient Score % Change
Group Name:Company Scorecard Details--Level Name:Overall Score Change YOY--Description:Overall Score Change YOY
Group Name:Company Scorecard Details--Level Name:Overall Score EOP--Description:Overall Score EOP
Group Name:Company Scorecard Details--Level Name:Overall Score Change YOY EOP--Description:Overall Score Change YOY EOP
Group Name:Company Scorecard Details--Level Name:Profit Score % Change EOP--Description:Profit Score % Change EOP
Group Name:Company Scorecard Details--Level Name:Efficient Score % Change EOP--Description:Efficient Score % Change EOP
Group Name:Company Scorecard Details--Level Name:Overall Score Trend--Description:Overall Score Trend
Group Name:Company Scorecard Details--Level Name:Coverage Score % Change EOP--Description:Coverage Score % Change EOP
Group Name:Interim Financial Details--Level Name:EBITDA Quarter--Description:EBITDA Quarter
Group Name:Interim Financial Details--Level Name:Net Sales Quarter--Description:Net Sales Quarter
Group Name:Interim Financial Details--Level Name:PAT--Description:PAT
Group Name:Interim Financial Details--Level Name:Corp Interest Coverage--Description:Corp Interest Coverage
Group Name:Budgets--Level Name:Average Balance Sum--Description:Budgeted Balance Amount - Sum
Group Name:Budgets--Level Name:Budgeted Revenue--Description:Budgeted Revenue Amount - Sum
Group Name:Budgets--Level Name:Budgeted Spread--Description:Budgeted Spread
Group Name:Budgets--Level Name:Budgeted EOP Balance--Description:Budgeted End of Period (EOP) Balance
Group Name:Budgets--Level Name:Budgeted Average Balance--Description:Budgeted Average Balance
Group Name:Customer Rating External--Level Name:Previous Rating Rank--Description:Previous Rating Rank
Group Name:Customer Rating External--Level Name:Rating Watch--Description:Rating Watch
Group Name:Customer Rating External--Level Name:Current Rating Rank--Description:Current Rating Rank
Group Name:Customer Rating External--Level Name:Rating Downgrade Notches--Description:Rating Downgrade Notches
Group Name:Customer Rating External--Level Name:Company Rating--Description:Company Rating
Group Name:Customer Rating External--Level Name:Change In External Rating--Description:Change In External Rating
Group Name:Customer Rating External--Level Name:Rating Agency count--Description:Rating Agency count
Group Name:Customer Rating External--Level Name:Internal Rating Notches--Description:Internal Rating Notches
Group Name:Transaction details--Level Name:Dummy_Inward_cheque--Description:Dummy_Inward_cheque
Group Name:Transaction details--Level Name:Dummy_outward_cheque--Description:Dummy_outward_cheque
Group Name:Transaction details--Level Name:Total Outward cheque--Description:Total Outward cheque
Group Name:Transaction details--Level Name:Bouncing Of High Value--Description:Bouncing Of High Value
Group Name:Transaction details--Level Name:Total Inward cheque--Description:Total Inward cheque
Group Name:Transaction details--Level Name:Bouncing Of Cheque--Description:Bouncing Of Cheque
Group Name:Transaction details--Level Name:return Date--Description:return Date
Group Name:Temp Measures--Level Name:Count of Products--Description:Count of Products
Group Name:Sanctions--Level Name:Disbursement Date--Description:Disbursement Date
Group Name:Sanctions--Level Name:Minus Regulization Date--Description:Minus Regulization Date
Group Name:Sanctions--Level Name:Regulization Date--Description:Regulization Date
Group Name:Sanctions--Level Name:Minus_Disb_Regul_Date--Description:Minus_Disb_Regul_Date
Group Name:Sanctions--Level Name:Disb_Regul_Date--Description:Disb_Regul_Date
Group Name:Sanctions--Level Name:Funding of Interest Sanction Additional Facility--Description:Funding of Interest Sanction Additional Facility
Group Name:Litigation Case Count--Level Name:Disposed Case--Description:Disposed Case
Group Name:Litigation Case Count--Level Name:Pending Case--Description:Pending Case
Group Name:Litigation Case Count--Level Name:Percentage Change Of New Litigation Case--Description:Percentage Change Of New Litigation Case
Group Name:Litigation Case Count--Level Name:Litigation Total Case Count--Description:Litigation Total Case Count
Group Name:Litigation Case Count--Level Name:No Of New Litigation Cases YOY--Description:No Of New Litigation Cases YOY
Group Name:Litigation Case Count--Level Name:No Of New Litigation Cases--Description:No Of New Litigation Cases
Group Name:Litigation Case Count--Level Name:No Of New Litigation Cases YOY EOP--Description:No Of New Litigation Cases YOY EOP
Group Name:Litigation Case Count--Level Name:No Of New Litigation Cases EOP--Description:No Of New Litigation Cases EOP
Group Name:Litigation Case Count--Level Name:Percentage Change Of New Litigation Case EOP--Description:Percentage Change Of New Litigation Case EOP
Group Name:Financial Interim Data--Level Name:Prev Interim Operating Margine--Description:Prev Interim Operating Margine
Group Name:Financial Interim Data--Level Name:Interim Interest Coverage Percentage--Description:Interim Interest Coverage Percentage
Group Name:Financial Interim Data--Level Name:Prev Interim Operating Margin On Year--Description:Prev Interim Operating Margin On Year
Group Name:Financial Interim Data--Level Name:Working Capital Day Percentage--Description:Working Capital Day Percentage
Group Name:Financial Interim Data--Level Name:Current Ratio on Year percentage--Description:Current Ratio on Year percentage
Group Name:Financial Interim Data--Level Name:Prev Working Capital Day--Description:Prev Working Capital Day
Group Name:Financial Interim Data--Level Name:Interim Working Capital Day--Description:Interim Working Capital Day
Group Name:Financial Interim Data--Level Name:Prev Interim Interest Coverage--Description:Prev Interim Interest Coverage
Group Name:Financial Interim Data--Level Name:Operating Margin On Year Percentage--Description:Operating Margin On Year Percentage
Group Name:Financial Interim Data--Level Name:Interim Current Ratio--Description:Interim Current Ratio
Group Name:Financial Interim Data--Level Name:Prev Interim Current Ratio--Description:Prev Interim Current Ratio
Group Name:Financial Interim Data--Level Name:Interim Gearing Times--Description:Interim Gearing Times
Group Name:Financial Interim Data--Level Name:Prev Interim Current Ratio on Year--Description:Prev Interim Current Ratio on Year
Group Name:Financial Interim Data--Level Name:Interim Operating Margine--Description:Interim Operating Margine
Group Name:Financial Interim Data--Level Name:Interim Current Ratio Percentage--Description:Interim Current Ratio Percentage
Group Name:Financial Interim Data--Level Name:Prev Interim Gearing Times--Description:Prev Interim Gearing Times
Group Name:Financial Interim Data--Level Name:Interim Operating Margine Percentage--Description:Interim Operating Margine Percentage
Group Name:Financial Interim Data--Level Name:Interim Interest Coverage--Description:Interim Interest Coverage
Group Name:Financial Interim Data--Level Name:Interest Coverage On Year Percentage--Description:Interest Coverage On Year Percentage
Group Name:Financial Interim Data--Level Name:Interim Gearing Times Percentage--Description:Interim Gearing Times Percentage
Group Name:Financial Interim Data--Level Name:Prev Interim Interest Coverage On Year--Description:Prev Interim Interest Coverage On Year
Group Name:Financial Interim Data--Level Name:Interest Coverage On Year EOP--Description:Interest Coverage On Year EOP
Group Name:Financial Interim Data--Level Name:Interim Current Ratio EOP--Description:Interim Current Ratio EOP
Group Name:Financial Interim Data--Level Name:Current Ratio On Year EOP--Description:Current Ratio On Year EOP
Group Name:Financial Interim Data--Level Name:Interest Coverage EOP--Description:Interest Coverage EOP
Group Name:Financial Interim Data--Level Name:Working Capital Days EOP--Description:Working Capital Days EOP
Group Name:Financial Interim Data--Level Name:Operating Margins EOP--Description:Operating Margins EOP
Group Name:Financial Interim Data--Level Name:Gearing Time EOP--Description:Gearing Time EOP
Group Name:Financial Interim Data--Level Name:Operating Margin On Year EOP--Description:Operating Margin On Year EOP
Group Name:Financial Tod Details--Level Name:TOD Advance Date Count--Description:TOD Advance Date Count
Group Name:Transactions--Level Name:Amount LCY--Description:Amount LCY
Group Name:Transactions--Level Name:Credit Transaction Amount--Description:Credit Transaction Amount
Group Name:Transactions--Level Name:Cash Credit Transaction Amount--Description:Cash Credit Transaction Amount
Group Name:Transactions--Level Name:Cash Credit Transaction Count--Description:Cash Credit Transaction Count
Group Name:Transactions--Level Name:Debit Transaction Count--Description:Debit Transaction Count
Group Name:Transactions--Level Name:Cash Debit Transaction Count--Description:Cash Debit Transaction Count
Group Name:Transactions--Level Name:Debit Transaction Amount--Description:Debit Transaction Amount
Group Name:Transactions--Level Name:Cash Debit Transaction Amount--Description:Cash Debit Transaction Amount
Group Name:Transactions--Level Name:Count Beneficiary code--Description:Count Beneficiary code
Group Name:Transactions--Level Name:Credit Transaction Count--Description:Credit Transaction Count
Group Name:Transactions--Level Name:Count AGG Acct Beneficiary code--Description:Count AGG Acct Beneficiary code
Group Name:Transactions--Level Name:Max RTGS To Sanction Amount--Description:Max RTGS To Sanction Amount
Group Name:Transactions--Level Name:High Value of RTGS Payments--Description:High Value of RTGS Payments
Group Name:Business Drivers--Level Name:Net Revenue--Description:Sum of Net Revenue of the customer/borrower
Group Name:Business Drivers--Level Name:Balance Amount Average--Description:Average balance amount of the customer/borrower
Group Name:Business Drivers--Level Name:CMS Volumes--Description:Cash Management Services (CMS) Volumes
Group Name:Business Drivers--Level Name:Gross Revenue--Description:Gross Revenue of the customer/ borrower
Group Name:Business Drivers--Level Name:Trade Income--Description:Trade Income of the customer / borrower
Group Name:Business Drivers--Level Name:Count of Customers--Description:Distinct count of Customers / borrowers
Group Name:Business Drivers--Level Name:CMS Income--Description:Income from Cash Management Services from the customer / borrower
Group Name:Business Drivers--Level Name:Fx Income--Description:Forex Income from the customer / borrower
Group Name:Business Drivers--Level Name:Total Income--Description:Total Income from the customer / borrower
Group Name:Business Drivers--Level Name:Unhedged Amount--Description:Unhedged Amount of the customer / borrower
Group Name:Business Drivers--Level Name:TP Rate--Description:Transfer Pricing (TP) Rate
Group Name:Business Drivers--Level Name:Unhedged FB Currency--Description:Balance Amount to Unhedged Amount Ratio
Group Name:Business Drivers--Level Name:Gross Rate--Description:Gross Rate
Group Name:Business Drivers--Level Name:Net Spread--Description:Net Spread
Group Name:Business Drivers--Level Name:Balance Amount--Description:End of Period (EOP) Balance Amount of the customer / borrower
Group Name:Business Drivers--Level Name:GL Balance--Description:End of Period (EOP) - General Ledger Balance
Group Name:Fraud Noticed--Level Name:Resignation Date--Description:Resignation Date
Group Name:Fraud Noticed--Level Name:Open Sarfaesi Case--Description:Open Sarfaesi Case
Group Name:Fraud Noticed--Level Name:Account Policy Change--Description:Account Policy Change
Group Name:Fraud Noticed--Level Name:Significant Inconsistency Count--Description:Significant Inconsistency Count
Group Name:Fraud Noticed--Level Name:Unfavourable CARO--Description:Unfavourable CARO
Group Name:Fraud Noticed--Level Name:Group Company Default Amount--Description:Group Company Default Amount
Group Name:Fraud Noticed--Level Name:Open Criminal Case--Description:Open Criminal Case
Group Name:Fraud Noticed--Level Name:Claims Not Ack Debit--Description:Claims Not Ack Debit
Group Name:Fraud Noticed--Level Name:Collateral Similarity--Description:Collateral Similarity
Group Name:Fraud Noticed--Level Name:DRT Pending Status--Description:DRT Pending Status
Group Name:Fraud Noticed--Level Name:Fraud Notice Count--Description:Fraud Notice Count
Group Name:Fraud Noticed--Level Name:Lending Total Open Charges--Description:Lending Total Open Charges
Group Name:Fraud Noticed--Level Name:Open Cheque Bounce Case--Description:Open Cheque Bounce Case
Group Name:Fraud Noticed--Level Name:Financial Report Qualified--Description:Financial Report Qualified
Group Name:Financial Balances--Level Name:Overdue Date--Description:Overdue Date
Group Name:Financial Balances--Level Name:Fb Remaining Outstanding--Description:Fb Remaining Outstanding
Group Name:Financial Balances--Level Name:Sanction Limit Sum--Description:Sanction Limit Sum
Group Name:Financial Balances--Level Name:Benificiary Amount LCY--Description:Benificiary Amount in Local Currency (LCY)
Group Name:Financial Balances--Level Name:Overdue_M2_01--Description:Overdue_M2_01
Group Name:Financial Balances--Level Name:Overdue_M5--Description:Overdue_M5
Group Name:Financial Balances--Level Name:Max Overdue Days Count--Description:Max Overdue Days Count
Group Name:Financial Balances--Level Name:Overdue_M1--Description:Overdue Month 1 (M1)
Group Name:Financial Balances--Level Name:Utilization Amount--Description:Utilization Amount
Group Name:Financial Balances--Level Name:Overdue_M2--Description:Overdue Month 2 (M2)
Group Name:Financial Balances--Level Name:Overdue_M5_01--Description:Overdue Month 5 (M5)
Group Name:Financial Balances--Level Name:Overdue_M3--Description:Overdue Month 3 (M3)
Group Name:Financial Balances--Level Name:Average Utilization--Description:Average Utilization
Group Name:Financial Balances--Level Name:Overdue_M4--Description:Overdue Month 4 (M4)
"""
        self.dim_measure = DimensionMeasure(self.dimensions, self.measures,self.llm,self.embedding)


        self.dim_measure.create_dimensions_db(self, cube_id)
        self.dim_measure.create_measures_db(self, cube_id)



from fastapi import APIRouter, FastAPI, Header, Request #BackgroundTasks
from fastapi.encoders import jsonable_encoder
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
from typing import Annotated
import jwt

DATA_FOLDER = "./datafolder/json/"
os.makedirs(DATA_FOLDER, exist_ok=True)

class RequestModel(BaseModel):
    # request input json structure
    user_query : str
    cube_id : int

logger = logging.getLogger('olap_genai')
logger.setLevel(logging.INFO)


app = FastAPI()
router = APIRouter()



@router.get("/genai/cube_query_generation")
async def root(request: Request, info:RequestModel):
    token = request.headers.get('authorization').split(" ")[1]
    print(token)
    if token: 
       try: 
           payload = jwt.decode(token, options={"verify_signature": False})
           print(payload)
           user_name = payload.get("preferred_username")
           return {"message": f"Hello, {user_name}!"} 
       except Exception as e: 
           return {"message": "Invalid token"} 

    mylist = []
    mylist.append(token)
    mylist.append(info)
    return mylist


@router.get("/genai/cube_details_import")
async def root(request: Request, info:RequestModel):
    token = request.headers.get('authorization').split(" ")[1]
    if token: 
       try: 
           payload = jwt.decode(token, options={"verify_signature": False})
           user_details = payload.get("preferred_username")
           return {"message": f"Hello, {user_details}!"} 
       except Exception as e: 
           return {"message": "Invalid token"} 
    mylist = []
    mylist.append(token)
    mylist.append(info)
    return mylist



app.include_router(router)

if __name__ == '__main__':
    uvicorn.run("fastapi_code 2:app", host="0.0.0.0", port=8085, workers=4)



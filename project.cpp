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
from typing import Dict, List, Tuple ,Optional
import os
import requests
from datetime import date
from colorama import Fore, Style ,init
from langchain_core.documents import Document
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory



def setup_logging():
    """
    function responsible for storing errors in log folder datewise and also stores token consumptions.
    """
    today = date.today()
    log_folder = './log'
  
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    logging.basicConfig(filename = f"{log_folder}/{today}.log",level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')


class LLMConfigure:
    """
    Class responsible for loading and configuring LLM and embedding models from a config file.
    """

    def __init__(self, config_path: json = "text2sql\config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            print(config_path)
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
            # LLM initialization using the config
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
            # embedding initialization using the config
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

    def __init__(self,llm: str,embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore
        self.vector_embedding= vectorstore
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 5

    def update_history(self,query,dimensions,measures, response):
        self.prev_query.append(query)
        self.prev_dimension.append(dimensions)
        self.prev_measures.append(measures)
        self.prev_response.append(response)

        self.prev_query=self.prev_query[-self.max_history:]
        self.prev_dimension= self.prev_dimension[-self.max_history:]
        self.prev_measures=self.prev_measures[-self.max_history:]
        self.prev_response=self.prev_response[-self.max_history:]


    def get_dimensions(self, query: str) -> str:
        """Extracts dimensions from the query."""
        try:

            with get_openai_callback() as dim_cb:
                
                query_dim = f""" 
                As an OLAP query expert, analyze the user's question and identify all required dimension information.
                
                User Query: {query}

                Below is the previous conversation details:-
                Previous Query:{self.prev_query}
                Previous Dimensions:{ self.prev_dimension}
                previous Response:{self.prev_response}

                <conversation_context>
                - Consider if this is a follow-up question that might reference previous dimensions implicitly
                - Look for temporal references like "last year", "previous month", etc.
                - Check for comparative references like "same as before but for..."
                - Identify any implicit dimensions that might be carried over from context
                </conversation_context>

                <instructions>
                - Select appropriate dimension group names and dimension level names strictly from context
                - Include all dimensions needed for accurate query generation
                - Handle both explicit and implicit dimension references
                - Consider temporal context for time-based dimensions
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["group1", "group2"],
                    "dimension_level_names": ["level1", "level2"],
                }}
                '''
                </instructions>
                """


                print(Fore.RED + '    Identifying Dimensions group name and level name......................\n')
                
                # NOTE: ---------------- uncomment below code only once if want to store vector embeddings of dimension table -------------------
                
                # vectordb = Chroma.from_documents(
                # documents=text_list_dim , # chunks
                # embedding=self.embedding, # instantiated embedding model
                # persist_directory=persist_directory # directory to save the data
                # )
                
            
                persist_directory_dim = self.vector_embedding['vector_embedding_path']['dimensions']
                load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=self.embedding)

                retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 50})
                ###########################3333
                # QA_CHAIN_PROMPT = PromptTemplate.from_template(query_dim)
                # # Run chain

                # qa = RetrievalQA.from_chain_type(
                #         self.llm,
                #         retriever=retriever_dim,
                #         return_source_documents=True,

                #         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                #     )
                            
                # print(qa)
                # result = qa({"query": query, "context":retriever_dim})
                # print(result)
                # response = result['result']
                # print(response)
                ###############################333333333
                
                chain_dim = RetrievalQA.from_chain_type(
                                                        llm = self.llm, 
                                                        retriever=retriever_dim,
                                                        verbose = True,
                                                        return_source_documents = True
                                                        )
                result_dim = chain_dim.invoke({"query": query_dim})
                dim = result_dim.get('result')
                
            
                logging.info(" ************* Token Consumed to Retrieve Dimensions**************************\n\n")
                logging.info(f"token consumed : {dim_cb.total_tokens}")
                logging.info(f"prompt tokens : {dim_cb.prompt_tokens}")
                logging.info(f"completion token : {dim_cb.completion_tokens}")
                logging.info(f"Total cost (USD) : {dim_cb.total_cost}")

                print(Fore.GREEN + '    Identified Group and level name :        ' + str(dim))
                logging.info(f"Extracted dimensions :\n {dim}")
                return dim
        
        except Exception as e:
            logging.error(f"Error extracting dimensions : {e}")
            raise
    
    def update_history(self,query,dimensions,measures, response):
        self.prev_query.append(query)
        self.prev_dimension.append(dimensions)
        self.prev_measures.append(measures)
        self.prev_response.append(response)

        self.prev_query=self.prev_query[-self.max_history:]
        self.prev_dimension= self.prev_dimension[-self.max_history:]
        self.prev_measures=self.prev_measures[-self.max_history:]
        self.prev_response=self.prev_response[-self.max_history:]
            

    def get_measures(self, query: str) -> str:
        """Extracts measures from the query."""
        try:
            
            with get_openai_callback() as msr_cb:
                
                 query_msr = f""" 
                As an OLAP query expert, analyze the user's question and identify all required measure information.
                
                User Query: {query}

                Below is the previous conversation details:-
                Previous Queries:{self.prev_query}
                Previous Measures:{self.prev_measures}
                previous Response:{self.prev_response}

                <conversation_context>
                - Consider if this is a follow-up question that might reference previous measures
                - Look for comparative phrases like "compare with", "versus", "instead of"
                - Check for aggregate references like "total", "average", "change in"
                - Identify any implicit measures from conversational context
                </conversation_context>

                <instructions>
                - Select appropriate measure group names and level names strictly from context
                - Include all measures needed for comprehensive analysis
                - Handle both explicit and implicit measure references
                - Consider aggregation requirements
                
                Response format:
                '''json
                {{
                    "measure_group_names": ["group1", "group2"],
                    "level_names": ["level1", "level2"],
                }}
                '''
                </instructions>
                """


            print(Fore.RED + '    Identifying Measure group name and level name......................\n')
                
                # NOTE: uncomment below lines only if want to store vector embedding of measure table for first time  ------
                
                # vectordb = Chroma.from_documents(
                # documents=text_list_msr , # chunks
                # embedding=self.embedding, # instantiated embedding model
                # persist_directory=persist_directory_msr # directory to save the data
                # )
                
            persist_directory_msr = self.vector_embedding['vector_embedding_path']['measures']
            load_embedding_msr = Chroma(persist_directory=persist_directory_msr, embedding_function=self.embedding)
            retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 60})
            chain_msr = RetrievalQA.from_chain_type(
                                                    llm = self.llm,
                                                    retriever = retriever_msr,
                                                    verbose=True,
                                                    return_source_documents=True
                                                    )
            print(chain_msr)
            result_msr = chain_msr.invoke({"query": query_msr})
            print(result_msr)
            msr = result_msr.get('result')
                
            print(f"Extracted Measure : {msr}")
            logging.info(f"Extracted Measure : {msr}")
            logging.info(" ******************** Token Consumed to Retrieve Measures **************************\n\n")
            logging.info(f"token consumed : {msr_cb.total_tokens}")
            logging.info(f"prompt tokens : {msr_cb.prompt_tokens}")
            logging.info(f"completion token : {msr_cb.completion_tokens}")
            logging.info(f"Total cost (USD) : {msr_cb.total_cost}")

            print(Fore.GREEN + '    Identified Group and level name :        ' + str(msr))
                
            return msr
        
        except Exception as e:
            print("Error:{}".format(e))
            logging.error(f"Error Extracting Measure")
        

class FinalQueryGenerator(LLMConfigure):
    """
    Class responsible for generating the final OLAP query based on dimensions and measures.
    """
    def __init__(self,query, dimensions: None, measures: None,llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 5

    def update_history(self,query,dimensions,measures, response):
        self.prev_query.append(query)
        self.prev_dimension.append(dimensions)
        self.prev_measures.append(measures)
        self.prev_response.append(response)

        self.prev_query=self.prev_query[-self.max_history:]
        self.prev_dimension= self.prev_dimension[-self.max_history:]
        self.prev_measures=self.prev_measures[-self.max_history:]
        self.prev_response=self.prev_response[-self.max_history:] 
        
    def call_gpt(self,final_prompt:str):
      """
      function responsible for generating final query
      """
      API_KEY = self.config['llm']['OPENAI_API_KEY']
      headers = {
          "Content-Type": "application/json",
          "api-key": API_KEY,
      }
      
      # Payload for the request
      payload = {
        "messages": [
          {
            "role": "system",
            "content": [
              {
                "type": "text",
                "text": "You are an AI assistant that writes accurate OLAP cube queries based on given query."
              }
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": final_prompt
              }
            ]
          }
        ],
        "temperature": self.config['llm']['temperature'],
        "top_p": self.config['llm']['top_p'],
        "max_tokens": self.config['llm']['max_tokens']
      }
      
      
      ENDPOINT = self.config['llm']['ENDPOINT']
      
      # Send request
      try:
          response = requests.post(ENDPOINT, headers=headers, json=payload)
          response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
      except requests.RequestException as e:
          raise SystemExit(f"Failed to make the request. Error: {e}")
      
      output = response.json()
      token_details = output['usage']
      output = output["choices"][0]["message"]["content"]
      return output ,token_details


    def generate_query(self, query: str, dimensions: str, measures: str, is_follow_up: bool) -> str:

        try:
            if not dimensions or not measures:
                raise ValueError("Both dimensions and measures are required to generate a query.")
                
            final_prompt = f"""Generate a precise OLAP cube query based on the following inputs and requirements.

            Input Context:
            - User Query: {query}
            - Dimensions: {dimensions}
            - Measures: {measures}
            - Is Follow-up:{is_follow_up}
            

            previous context:-
            - Previous Dimensions: {self.prev_dimension}
            - Previous Measures: {self.prev_measures}
            - Previous User Query: {self.prev_query}
            - Previous Cube Query: {self.prev_response}

            <conversational_guidelines>
            - for follow-ups, maintain consistencywith previous elements unless changed
            - Handle implicit references to previous elements 
            - Preserve specific aggregation/calculation from context
            - Maintain temporal context("last year","previous quarter")
            - Consider explicit and implicit filters from conversation
            </conversational_guidelines>

            Requirements:
            1. Generate a single-line OLAP query without line breaks
            2. Always start with 'select' followed by dimensions and measures
            3. Always use the exact cube name: [Cube].[Credit One View]
            4. Include 'as' aliases for all columns in double quotes
            5. Use proper functions based on query requirements:
            - TimeBetween for date ranges
            - TrendNumber for year-over-year/period comparisons
            - Head/Tail for top/bottom N queries
            - RunningSum/PercentageOfRunningSum for cumulative calculations

            Formatting Rules:
            1. Dimensions format: [Dimension Group].[Level] as "Alias"
            2. Measures format: [Measure Group].[Measure] as "Alias"
            3. Conditions in WHERE clause must be properly formatted with operators
            4. For multiple conditions, use 'and'/'or' operators
            5. All string values in conditions must be in single quotes
            6. All numeric values should not have leading zeros

            Example Queries:
            1. Basic selection:
            select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit One View]

            2. With condition:
            select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit One View] where [Business Drivers].[Count of Customers] > 10.00

            3. With time trends:
            select [Time].[Year] as "Year", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", TrendNumber([Business Drivers].[Balance Amount Average],[Time].[Year],1,'percentage') as "YOY % Change" from [Cube].[Credit One View]

            4. With date range:
            select [Time].[Year] as "Year", [Share Price BSE].[Close Price] as "Close Price", TimeBetween(20160101,20191231,[Time].[Year], false) from [Cube].[Credit One View]

            <examples>

            Q1. Which months has the value of balance average amount between 40,00,00,000 to 2,00,00,00,000 ?           
            select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit one View] where [Business Drivers].[Balance Amount Average] 400000000.00 between 2000000000.00

            Q2.Top 5 Cities on Average Balance Amount.
            select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average",Head([Branch Details].[City],[Business Drivers].[Balance Amount Average],5,undefined) from [Cube].[Credit one View]

            Q3.Please provide Cities and Average Balance Amount where Average Balance Amount is more than 5000 and Count of Customers more than 10.
            select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit one View] where [Business Drivers].[Count of Customers] > 10.00 and [Business Drivers].[Balance Amount Average] > 5000.00

            Q4.Please provide Year and Average Balance Amount and % change from previous year.
            select [Time].[Year] as "Year", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", TrendNumber([Business Drivers].[Balance Amount Average],[Time].[Year],1,'percentage') as "Balance Amount YOY %" from [Cube].[Credit one View]

            Q5.Please provide Year and Average Balance Amount and % change from previous year for past 2 years
            select [Time].[Year] as "Year", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", TrendNumber([Business Drivers].[Balance Amount Average],[Time].[Year],1,'percentage') as "YOY % Change",TimeBetween(20200101,20231219,[Time].[Year], false) from [Cube].[Credit one View]

            Q4.What is the closing price of Industry ?
            select [Industry Details].[Industry Name] as "Industry Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[Credit one View]

            Q5.Which are the bottom 4 years having lowest total revenue from operation ?
            select [Time].[Year] as "Year", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",Tail([Time].[Year],[Financial Data].[Total Revenue From Operations],4,undefined) from [Cube].[Credit one View]

            Q6.What are the closing price of AXIS, HDFC, ICICI & LIC Mutual Funds ?
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[Credit one View] where [Mutual Fund Investment].[Mutual Fund Name] in ('Axis','HDFC','ICICI','LIC')

            Q7.What are the Cash Ratio Current Ratio and Quick Ration Based on Month from 2012 to 2017 ? 
            select [Time].[Month] as "Month", [Financial Ratios].[Cash Ratio] as "Cash Ratio", [Financial Ratios].[Current Ratio] as "Current Ratio", 
            [Financial Ratios].[Quick Ratio] as "Quick Ratio",TimeBetween(20120101,20171231,[Time].[Year], false) from [Cube].[Credit one View]

            Q8.What are the closing Price of Share from Year 2016 to 2019 ?
            select [Time].[Year] as "Year", [Share Price BSE].[Close Price] as "Close Price",
            TimeBetween(20160101,20191231,[Time].[Year], false) from [Cube].[Credit one View]

            Q9.Provide the Year and Total Revenue and Value of Previous YearA21.
            select [Time].[Year] as "Year", [Financial Data].[Total Revenue] as "Total Revenue", 
            TrendNumber([Financial Data].[Total Revenue],[Time].[Year],1,'value') as "Trend 1" from [Cube].[Credit one View]

            Q10.Provide the Quarter wise %Change from Previous year for Total Revenue.
            select [Time].[Quarter] as "Quarter", [Financial Data].[Total Revenue] as "Total Revenue", TrendNumber([Financial Data].[Total Revenue],[Time].[Quarter],1,'percentage') as "Trend" from [Cube].[Credit one View]

            Q11.What are the Bottom 4 state based on Total Debit ?
            select [Branch Details].[State] as "State", [Financial Data].[Total Debit] as "Total Debit",Tail([Branch Details].[State],[Financial Data].[Total Debit],4,undefined) from [Cube].[Credit one View]

            Q12.Which Mutual Funds have the trade price greater than 272 but less than 276 ?
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Bulk Deal Trade].[Trade Price] as "Trade Price" from [Cube].[Credit one View] where [Bulk Deal Trade].[Trade Price] < 276.00 and [Bulk Deal Trade].[Trade Price] > 272.00

            Q13.What is the Balance Amount for Mutual funds other than HDFC,SBI,Nippon,HSBC,ICICI,IDFC,AXIS ?
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[Credit one View] where [Mutual Fund Investment].[Mutual Fund Name] not in ('SBI','Nippon','HDFC','HSBC','ICICI','IDFC','Axis')

            Q14.Provide the Mutual Fund and Balance Amount.
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[Credit one View]

            Q15.Provide the Running sum of Balance amount with Mutual Fund.
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", runningsum([Business Drivers].[Balance Amount],'sumacrossrows') as "Running Sum Balance Amount" from [Cube].[Credit one View]

            Q16.What is the % of Running Sum of Balance amount with respect to Mutual Fund.
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", 
            percentageofrunningsum([Business Drivers].[Balance Amount],'percentagerunningsumacrossrows') as "% of Running sum of Balance Amount" from [Cube].[Credit one View]
                        
            Q17.Provide the Close price of Index Nifty and Sensex based on Index Name that doesn't contains "Nifty".
            select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[Credit one View] where [Benchmark Index Details].[Index Name] not like '%Nifty%'. 

            Q18.Provide the list of months not having the balance average amount between 40,00,00,000 to 2,00,00,00,000.               
            select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit one View] where [Business Drivers].[Balance Amount Average] 400000000.00 not between 2000000000.00

            Q19.Provide the Month wise Balance Amount with Mutual Fund Name with Balance Amount.
            select [Time].[Month] as "Month", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[Credit one View]

            Q20.Provide the Close price of Index Nifty and Sensex based on Index Name that contains "Nifty".
            select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[Credit one View] where [Benchmark Index Details].[Index Name] like '%Nifty%' 

            Q21.Show me the Customer Name and Mutual Fund Name with Balance Amount and Balance Average Amount.
            select [Customer Details].[Customer Name] as "Customer Name", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit one View]

            Q22.Show me the Customer Name and Mutual Fund Name with Balance Amount Greater than 0 and Balance Average Amount with % of Balance Amount.
            select [Customer Details].[Customer Name] as "Customer Name", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", percentage([Business Drivers].[Balance Amount],'percentColumn') as "% of Balance Amount" from [Cube].[Credit one View] where [Business Drivers].[Balance Amount] > 0.00

            Q23.Please provide zone and city wise Average Balance Amount.
            select [Branch Details].[Zone] as "Zone", [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[Credit one View]

            Q24.Provide the Mutual Fund and their Quantity along with Month on Month Quantity and Quarter on Quarter Quantity.
            select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Fund Investment Details].[Mutual Fund Quantity] as "Mutual Fund Quantity", [Fund Investment Details].[Mutual Fund Quantity MoM] as "Mutual Fund Quantity MoM", [Fund Investment Details].[Mutual Fund Quantity QoQ] as "Mutual Fund Quantity QoQ" from [Cube].[Credit one View]
            
            <response_format>
            Follow-up Query Handling:
            1. Maintain consistency with previous dimensions unless explicitly changed
            2. Preserve temporal context from previous queries
            3. Handle implicit references to previous measures
            4. Maintain aggregation levels from previous context
            5. Consider filters from previous query when relevant
            6. If the change is done in measures give same dimension with given measure
            7. If only 3-4 words given with change in measures give the last query all dimensions strictly.
            <examples>
            1.Which are the bottom 4 years having lowest total revenue from operation ?
            select [Time].[Year] as "Year", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",Tail([Time].[Year],[Financial Data].[Total Revenue From Operations],4,undefined) from [Cube].[Credit one View]

            2.bottom 2 years
            select [Time].[Year] as "Year", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",Tail([Time].[Year],[Financial Data].[Total Revenue From Operations],2,undefined) from [Cube].[Credit one View]
            
            3.What are the closing Price of Share from Year 2016 to 2019 ?
            select [Time].[Year] as "Year", [Share Price BSE].[Close Price] as "Close Price",
            TimeBetween(20160101,20191231,[Time].[Year], false) from [Cube].[Credit one View]

            4.Year 2018 to 2022?
                select [Time].[Year] as "Year", [Share Price BSE].[Close Price] as "Close Price",
            TimeBetween(20180101,20221231,[Time].[Year], false) from [Cube].[Credit one View]

            Generate a precise single-line OLAP query that exactly matches these requirements:"""                
            print(Fore.CYAN + '   Generating OLAP cube Query......................\n')
            #output,token_details= self.call_gpt(final_prompt)
            result = self.llm.invoke(final_prompt)
            output = result.content
            token_details = result.response_metadata['token_usage']
            pred_query = self.cleanup_gen_query(output)
            print(f"{pred_query}")
            
            logging.info(f"Generated OLAP Query: {pred_query}")
            logging.info(" ************ token consumed to return final OLAP query ****************\n\n")
            logging.info(f"completion_tokens : {token_details['completion_tokens']}")
            logging.info(f"prompt_tokens : {token_details['prompt_tokens']}")
            logging.info(f"total_tokens : {token_details['total_tokens']}")
                            
            return Fore.BLUE + output
        
        except Exception as e:
            logging.error(f"Error generating OLAP query: {e}")
            raise
    
    def cleanup_gen_query(self,pred_query):
        
        check = pred_query.replace("```","")
        final_query = check.replace("sql","")
        return final_query



class QueryContext:
    """Store context for each query"""
    def __init__(self,
        query: str = None,
        dimensions: Dict=None,
        measures: Dict=None,
        olap_query: str=None,
        timestamp: float =None,
        context_type: str = None,  
        parent_query: Optional[str] = None):
        self.query= query # type: ignore
        self.dimensions= dimensions
        self.measures=measures
        self.olap_query=olap_query
        self.timestamp=timestamp
        self.context_type=context_type 
        self.parent_query=parent_query

class ConversationManager:
    """Manages conversation context and follow-up detection"""
    def __init__(self):
        self.context_window = []  # Stores recent queries with context
        self.max_context_size = 5
        
    def add_context(self, query_context: QueryContext):
        self.context_window.append(query_context)
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def get_recent_context(self) -> List[QueryContext]:
        return self.context_window
    
    def is_follow_up(self, query: str) -> bool:
        """Detect if query is a follow-up based on various indicators"""
        if not self.context_window:
            return False
            
        follow_up_indicators = [
            "this", "that", "these", "those", "it", "previous",
            "instead", "but", "also", "too", "same", "like before",
            "show me", "give me", "what about", "how about"
        ]
        
        query_lower = query.lower()
        # Check for numeric-only queries (like "show me 10")
        if query_lower.replace("show me", "").replace("give me", "").strip().isdigit():
            return True
            
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
                
        # Check if query is very short (likely a follow-up)
        if len(query.split()) <= 3:
            return True
            
        return False

class OLAPQueryProcessor(LLMConfigure):
    """Enhanced OLAP processor with conversation memory"""
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.conversation_manager = ConversationManager()
        self.memory = ConversationBufferMemory(return_messages=True)

        try:
            # Initialize other components
            print("config_path in cube_olap=", config_path)
            self.llm_config = LLMConfigure(config_path)
            self.load_json = self.llm_config.load_config(config_path)
            self.llm = self.llm_config.initialize_llm()
            self.embedding = self.llm_config.initialize_embedding()
            self.dim_measure = DimensionMeasure(self.llm, self.embedding, self.load_json)
            self.final = FinalQueryGenerator(query="", dimensions=None, measures=None, llm=self.llm)
            self.query_history = []

        except Exception as e:
            logging.error(f"Error initializing EnhancedOLAPQueryProcessor: {e}")
            raise

    def process_query(self, query: str) -> Tuple[str, str, float]:
        """Process a query with timing and context handling"""
        print(f"In process query :- User query : {query} ")
        try:
            start_time = time.time()
            dimensions = self.dim_measure.get_dimensions(query)
            measures = self.dim_measure.get_measures(query)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures")

            is_follow_up = self.conversation_manager.is_follow_up(query)
            final_query = self.final.generate_query(query, dimensions, measures, is_follow_up)
            
            # Save context
            context = QueryContext(
                query=query,
                dimensions=dimensions,
                measures=measures,
                olap_query=final_query,
                timestamp=time.time(),
                context_type='follow_up' if is_follow_up else 'initial'
            )

            self.conversation_manager.add_context(context)
            self.dim_measure.update_history(query, dimensions, measures, final_query)
            self.final.update_history(query, dimensions, measures, final_query)
            
            processing_time = time.time() - start_time
            return query, final_query, processing_time,dimensions,measures
            
        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            raise

def main():
    """Enhanced main function with better conversation handling"""
    setup_logging()
    config_path = "text2sql\config.json"
    
    processor = OLAPQueryProcessor(config_path)
    
    print(Fore.CYAN + "\n=== OLAP Query Conversation System ===")
    print(Fore.CYAN + "Type 'exit' to end the conversation.\n")
    
    while True:
        try:
            query = input(Fore.GREEN + "Please enter your query: ")
            
            if query.lower() == 'exit':
                print(Fore.YELLOW + "\nThank you for using the OLAP Query System! Goodbye!")
                break
            
            # Process query with enhanced context handling
            original_query, final_query, processing_time,dimensions,measures = processor.process_query(query)            
            print(Fore.CYAN + f"\nProcessing time: {processing_time:.2f} seconds\n")
            print()  # Add spacing for readability

 
                               
        except Exception as e:
            logging.error(f"Error in conversation: {e}")
            print(Fore.RED + f"\nI encountered an error: {str(e)}")
            print(Fore.YELLOW + "Please try rephrasing your question or ask something else.\n")
            continue


if __name__ == "__main__":
    main()
  

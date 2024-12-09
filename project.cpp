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
}# Add these imports if not already present
from typing import Literal, Optional
from pydantic import BaseModel

class UserFeedbackRequest(BaseModel):
    user_feedback: str 
    feedback: Literal["accepted", "rejected"]
    cube_query: str
    cube_id: str

class UserFeedbackResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None

@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(
    request: UserFeedbackRequest,
    user_details: str = Depends(verify_token)
):
    """Handle user feedback for cube queries"""
    try:
        user_id = f"user_{user_details}"
        
        if request.feedback == "rejected":
            # Get or create processor
            if user_id not in olap_processors:
                olap_processors[user_id] = OLAPQueryProcessor(config_file)
            
            processor = olap_processors[user_id]
            history_manager = History()
            prev_conv = history_manager.retrieve(user_id)
            
            # Add feedback to context
            prev_conv["user_feedback"] = request.user_feedback
            prev_conv["feedback_query"] = request.cube_query
            
            # Process query with feedback context
            query, final_query, _, dimensions, measures = processor.process_query(
                request.user_feedback,
                request.cube_id, 
                prev_conv
            )
            
            # Update history
            response_data = {
                "query": request.user_feedback,
                "dimensions": dimensions,
                "measures": measures,
                "response": final_query
            }
            history_manager.update(user_id, response_data)
            
            return UserFeedbackResponse(
                message="success",
                cube_query=final_query
            )
            
        return UserFeedbackResponse(
            message="success", 
            cube_query=request.cube_query
        )
        
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return UserFeedbackResponse(
            message="failure",
            cube_query=None
        )




  def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extract dimensions with feedback consideration"""
    try:
        with get_openai_callback() as dim_cb:
            # Add feedback context if exists
            feedback_section = ""
            if "user_feedback" in prev_conv:
                feedback_section = f"""
                Previous User Feedback: {prev_conv['user_feedback']}
                Feedback Query: {prev_conv['feedback_query']}
                """

            query_dim = f""" 
            As an query to cube query convertion expert, analyze the user's question and identify all required dimension information.
            
            User Query: {query}

            Below is the previous conversation details:-
            Previous Query: {prev_conv["query"]}
            Previous Dimensions: {prev_conv["dimensions"]}
            Previous Cube Query: {prev_conv["response"]}
            {feedback_section}

            <conversation_context>
            - Consider if this is a follow-up question that might reference previous dimensions implicitly
            - Look for temporal references like "last year", "previous month", etc.
            - Check for comparative references like "same as before but for..."
            - Identify any implicit dimensions that might be carried over from context
            </conversation_context>

            Response format:
            '''json
            {
                "dimension_group_names": ["group1", "group2"],
                "dimension_level_names": ["level1", "level2"]
            }
            '''
            """

            cube_dir = os.path.join(vector_db_path, cube_id)
            cube_dim = os.path.join(cube_dir, "dimensions")
            load_embedding_dim = Chroma(persist_directory=cube_dim, embedding_function=self.embedding)
            
            retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 50})
            chain_dim = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever_dim,
                verbose=True,
                return_source_documents=True
            )
            
            result_dim = chain_dim.invoke({"query": query_dim})
            dimensions = result_dim.get('result')
            
            return dimensions

    except Exception as e:
        logging.error(f"Error extracting dimensions: {e}")
        raise

def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extract measures with feedback consideration"""
    try:
        with get_openai_callback() as msr_cb:
            # Add feedback context if exists  
            feedback_section = ""
            if "user_feedback" in prev_conv:
                feedback_section = f"""
                Previous User Feedback: {prev_conv['user_feedback']}
                Feedback Query: {prev_conv['feedback_query']}
                """

            query_msr = f""" 
            As an OLAP query expert, analyze the user's question and identify all required measure information.
            
            User Query: {query}

            Below is the previous conversation details:-
            Previous Query: {prev_conv["query"]}
            Previous Measures: {prev_conv["measures"]}
            Previous Cube Query: {prev_conv["response"]}
            {feedback_section}

            Response format:
            '''json
            {
                "measure_group_names": ["group1", "group2"],
                "measure_names": ["measure1", "measure2"]
            }
            '''
            """

            cube_dir = os.path.join(vector_db_path, cube_id)
            cube_msr = os.path.join(cube_dir, "measures")
            load_embedding_msr = Chroma(persist_directory=cube_msr, embedding_function=self.embedding)
            
            retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 60})
            chain_msr = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever_msr,
                verbose=True,
                return_source_documents=True
            )
            
            result_msr = chain_msr.invoke({"query": query_msr})
            measures = result_msr.get('result')
            
            return measures

    except Exception as e:
        logging.error(f"Error extracting measures: {e}")
        raise

def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict) -> str:
    """Generate query with feedback consideration"""
    try:
        # Add feedback context if exists
        feedback_section = ""
        if "user_feedback" in prev_conv:
            feedback_section = f"""
            Previous User Feedback: {prev_conv['user_feedback']}
            Feedback Query: {prev_conv['feedback_query']}
            """

        final_prompt = f"""Generate a precise OLAP cube query based on the following inputs and requirements.

        Input Context:
        - User Query: {query}
        - Dimensions: {dimensions}
        - Measures: {measures}

        Previous Context:
        - Previous Query: {prev_conv["query"]}
        - Previous Dimensions: {prev_conv["dimensions"]}
        - Previous Measures: {prev_conv["measures"]}
        - Previous Cube Query: {prev_conv["response"]}
        {feedback_section}

        Requirements:
        1. Generate a single-line OLAP query without line breaks
        2. Always start with 'select' followed by dimensions and measures
        3. Always use the exact cube name: [Cube].[Credit One View]
        4. Include 'as' aliases for all columns in double quotes        
        """
        
        result = self.llm.invoke(final_prompt)
        output = result.content
        token_details = result.response_metadata['token_usage']
        pred_query = self.cleanup_gen_query(output)
        
        return output

    except Exception as e:
        logging.error(f"Error generating query: {e}")
        raise

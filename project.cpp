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
# Add these functions to cube_query_v3.py

class DimensionMeasure:
    # Add this method to existing DimensionMeasure class
    def get_dimensions_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str) -> str:
        """Extracts dimensions with user feedback consideration."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim_feedback = f"""
                As a user query to cube query conversion expert, analyze the user feedback for the cube query. 
                Your goal is to refine the identification of dimensions based on user feedback.
                
                Below details using which the feedback was received:-
                User Query: {query}
                Current Dimensions: {prev_conv["dimensions"]}
                Current Cube Query Response: {prev_conv["response"]}
                Feedback Type: {feedback_type}

                <feedback_analysis_context>
                - For accepted feedback: reinforce successful dimension mappings
                - For rejected feedback: identify potential dimension mismatches
                - Consider previous successful queries as reference
                - Analyze dimension hierarchy appropriateness
                - Check for implicit dimension references
                </feedback_analysis_context>

                <refinement_guidelines>
                - Strictly provide dimensions only
                - For accepted feedback: maintain successful mappings
                - For rejected feedback: suggest alternative dimensions
                - Consider dimension hierarchies
                - Maintain temporal relationships
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["refined_group1", "refined_group2"],
                    "dimension_level_names": ["refined_level1", "refined_level2"],
                    "refinements": ["Maintained dimension X", "Changed dimension Y"],
                    "reasoning": "Explanation of refinements made"
                }}
                '''
                </refinement_guidelines>
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
                result_dim = chain_dim.invoke({"query": query_dim_feedback})
                return result_dim.get('result')

        except Exception as e:
            logging.error(f"Error in dimension feedback processing: {e}")
            raise

    def get_measures_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str) -> str:
        """Extracts measures with user feedback consideration."""
        try:
            with get_openai_callback() as msr_cb:
                query_msr_feedback = f"""
                As a user query to cube query conversion expert, analyze the user feedback for the cube query. 
                Your goal is to refine the identification of measures based on user feedback.
                
                Below details using which the feedback was received:-
                User Query: {query}
                Current Measures: {prev_conv["measures"]}
                Current Cube Query Response: {prev_conv["response"]}
                Feedback Type: {feedback_type}

                <feedback_analysis_context>
                - For accepted feedback: reinforce successful measure mappings
                - For rejected feedback: identify potential measure mismatches
                - Consider aggregation appropriateness
                - Check for calculation errors
                - Analyze measure combinations
                </feedback_analysis_context>

                <refinement_guidelines>
                - Strictly provide measures only
                - For accepted feedback: maintain successful mappings
                - For rejected feedback: suggest alternative measures
                - Consider aggregation functions
                - Maintain calculation integrity
                
                Response format:
                '''json
                {{
                    "measure_group_names": ["refined_group1", "refined_group2"],
                    "measure_names": ["refined_measure1", "refined_measure2"],
                    "refinements": ["Maintained measure X", "Changed measure Y"],
                    "reasoning": "Explanation of refinements made"
                }}
                '''
                </refinement_guidelines>
                """

                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_msr = os.path.join(cube_dir, "measures")
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
                result_msr = chain_msr.invoke({"query": query_msr_feedback})
                return result_msr.get('result')

        except Exception as e:
            logging.error(f"Error in measure feedback processing: {e}")
            raise

class OLAPQueryProcessor:
    # Add this method to existing OLAPQueryProcessor class
    def process_query_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str) -> Tuple[str, str, float, Dict, Dict]:
        """Process a query with user feedback consideration."""
        try:
            start_time = time.time()
            # Get refined dimensions and measures based on feedback
            dimensions = self.dim_measure.get_dimensions_with_feedback(query, cube_id, prev_conv, feedback_type)
            measures = self.dim_measure.get_measures_with_feedback(query, cube_id, prev_conv, feedback_type)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures after feedback processing")

            final_query = self.final.generate_query(query, dimensions, measures, prev_conv)
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures

        except Exception as e:
            logging.error(f"Error in query processing with feedback: {e}")
            raise

# Add these models and endpoint to olap_details_generat.py

class UserFeedbackRequest(BaseModel):
    user_feedback: str
    feedback: Literal["accepted", "rejected"]
    cube_query: str
    cube_id: str

class UserFeedbackResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None

@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(request: UserFeedbackRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        history_manager = History()
        prev_conversation = history_manager.retrieve(user_id)
        
        query, final_query, processing_time, dimensions, measures = OLAPQueryProcessor(config_file).process_query_with_feedback(
            request.user_feedback,
            request.cube_id,
            prev_conversation,
            request.feedback
        )
        
        response_data = {
            "query": query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,
        }
        
        history_manager.update(user_id, response_data)
        
        return UserFeedbackResponse(
            message="success",
            cube_query=final_query
        )
    except Exception as e:
        logging.error(f"Error in handle_user_feedback: {e}")
        return UserFeedbackResponse(
            message="failure",
            cube_query=None
        )

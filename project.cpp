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
# Add to your cube_query_v3.py file

class DimensionMeasure:
    # Add these new methods to existing DimensionMeasure class
    
    def get_dimensions_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_data: dict) -> str:
        """Extracts dimensions taking into account user feedback."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim = f""" 
                As an query to cube query conversion expert, analyze the user's feedback and question to identify required dimension information.
                
                User Query: {query}
                User Feedback: {feedback_data['user_feedback']}
                Previous Query Response: {feedback_data['cube_query']}
                Feedback Type: {feedback_data['feedback']}

                Previous Conversation Context:
                Previous Query: {prev_conv["query"]}
                Previous Dimensions: {prev_conv["dimensions"]}
                Previous Cube Query Response: {prev_conv["response"]}

                <feedback_analysis>
                - Consider user's specific feedback about what was wrong
                - Analyze if dimensions need to be adjusted based on feedback
                - Check if user is requesting different granularity
                - Verify if temporal context needs adjustment
                - Look for specific mentions of dimension changes in feedback
                - Consider implicit dimension preferences from feedback
                </feedback_analysis>

                <response_guidelines>
                - Adjust dimension selections based on feedback
                - Maintain valid dimension combinations
                - Ensure proper hierarchy levels
                - Consider temporal context carefully
                - Validate against available dimensions
                </response_guidelines>

                Response format:
                '''json
                {
                    "dimension_group_names": ["group1", "group2"],
                    "dimension_level_names": ["level1", "level2"],
                    "feedback_adjustments": ["adjustment1", "adjustment2"]
                }
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
                dimensions = result_dim.get('result')
                
                logging.info(f"Extracted dimensions with feedback:\n {dimensions}")
                logging.info(" ************* Token Consumed to Retrieve Dimensions with Feedback **************")
                logging.info(f"token consumed : {dim_cb.total_tokens}")
                logging.info(f"prompt tokens : {dim_cb.prompt_tokens}")
                logging.info(f"completion token : {dim_cb.completion_tokens}")
                logging.info(f"Total cost (USD) : {dim_cb.total_cost}")
                
                return dimensions

        except Exception as e:
            logging.error(f"Error extracting dimensions with feedback: {e}")
            raise

    def get_measures_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_data: dict) -> str:
        """Extracts measures taking into account user feedback."""
        try:
            with get_openai_callback() as msr_cb:
                query_msr = f""" 
                As an OLAP query expert, analyze the user's feedback and question to identify required measure information.
                
                User Query: {query}
                User Feedback: {feedback_data['user_feedback']}
                Previous Query Response: {feedback_data['cube_query']}
                Feedback Type: {feedback_data['feedback']}

                Previous Conversation Context:
                Previous Query: {prev_conv["query"]}
                Previous Measures: {prev_conv["measures"]}
                Previous Cube Query Response: {prev_conv["response"]}

                <feedback_analysis>
                - Consider user's feedback about measure selection
                - Check if different aggregations are needed
                - Verify if calculations need adjustment
                - Analyze if measure combinations are appropriate
                - Look for specific mentions of measures in feedback
                - Consider implicit measure preferences
                </feedback_analysis>

                <response_guidelines>
                - Adjust measure selections based on feedback
                - Ensure proper aggregation functions
                - Validate calculation methods
                - Maintain measure compatibility
                - Check against available measures
                </response_guidelines>

                Response format:
                '''json
                {
                    "measure_group_names": ["group1", "group2"],
                    "measure_names": ["measure1", "measure2"],
                    "feedback_adjustments": ["adjustment1", "adjustment2"]
                }
                '''
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
                
                result_msr = chain_msr.invoke({"query": query_msr})
                measures = result_msr.get('result')
                
                logging.info(f"Extracted measures with feedback:\n {measures}")
                logging.info(" ************* Token Consumed to Retrieve Measures with Feedback **************")
                logging.info(f"token consumed : {msr_cb.total_tokens}")
                logging.info(f"prompt tokens : {msr_cb.prompt_tokens}")
                logging.info(f"completion token : {msr_cb.completion_tokens}")
                logging.info(f"Total cost (USD) : {msr_cb.total_cost}")
                
                return measures

        except Exception as e:
            logging.error(f"Error extracting measures with feedback: {e}")
            raise


class OLAPQueryProcessor:
    # Add this new method to existing OLAPQueryProcessor class
    
    def process_query_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_data: dict) -> Tuple[str, str, float, Dict, Dict]:
        """Process a query taking into account user feedback."""
        try:
            start_time = time.time()
            
            # Get dimensions and measures with feedback consideration
            dimensions = self.dim_measure.get_dimensions_with_feedback(query, cube_id, prev_conv, feedback_data)
            measures = self.dim_measure.get_measures_with_feedback(query, cube_id, prev_conv, feedback_data)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures with feedback")

            # Generate new query considering feedback
            final_query = self.final.generate_query(query, dimensions, measures, prev_conv)
            
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures

        except Exception as e:
            logging.error(f"Error in query processing with feedback: {e}")
            raise

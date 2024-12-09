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


def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extract dimensions with feedback consideration"""
    try:
        with get_openai_callback() as dim_cb:
            # Add feedback context if exists
            feedback_section = ""
            if "user_feedback" in prev_conv:
                feedback_section = f"""
                User Feedback Analysis:
                - Previous Query: {prev_conv['feedback_query']}
                - User's Feedback: {prev_conv['user_feedback']}

                <feedback_context>
                - Consider if feedback mentions missing or incorrect dimensions
                - Check if feedback suggests different granularity levels
                - Look for references to wrong hierarchies or dimension combinations
                - Analyze if temporal dimensions need adjustment
                - Consider if feedback implies need for additional filtering dimensions
                </feedback_context>
                """

            query_dim = f""" 
            As an query to cube query conversion expert, analyze the user's question and identify all required dimension information.
            
            User Query: {query}

            Previous Conversation Context:
            Previous Query: {prev_conv["query"]}
            Previous Dimensions: {prev_conv["dimensions"]}
            Previous Cube Query: {prev_conv["response"]}

            {feedback_section}

            <dimension_guidelines>
            - Select appropriate dimension group names and levels based on requirements
            - Ensure temporal dimensions match required granularity
            - Include necessary hierarchical dimensions
            - Consider geography and organizational hierarchies if relevant
            - Add any filtering dimensions needed for context
            </dimension_guidelines>

            <validation_rules>
            - Verify all dimensions exist in the cube structure
            - Check dimension hierarchy compatibility
            - Ensure temporal granularity matches requirements
            - Validate dimension combinations are valid
            </validation_rules>

            Response format:
            '''json
            {{
                "dimension_group_names": ["group1", "group2"],
                "dimension_level_names": ["level1", "level2"],
                "reasoning": ["reason for selecting each dimension"]
            }}
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
            feedback_section = ""
            if "user_feedback" in prev_conv:
                feedback_section = f"""
                User Feedback Analysis:
                - Previous Query: {prev_conv['feedback_query']}
                - User's Feedback: {prev_conv['user_feedback']}

                <feedback_context>
                - Check if feedback mentions incorrect measures
                - Look for references to wrong calculations or aggregations
                - Consider if additional measures are needed
                - Analyze if measure combinations need adjustment
                - Verify if user wants different measure formats
                </feedback_context>
                """

            query_msr = f""" 
            As an OLAP query expert, analyze the user's question and identify all required measure information.
            
            User Query: {query}

            Previous Conversation Context:
            Previous Query: {prev_conv["query"]}
            Previous Measures: {prev_conv["measures"]}
            Previous Cube Query: {prev_conv["response"]}

            {feedback_section}

            <measure_guidelines>
            - Select appropriate measure groups based on analysis needs
            - Choose correct aggregation functions
            - Include calculated measures if needed
            - Consider measure combinations for analysis
            - Add trending or comparison measures if relevant
            </measure_guidelines>

            <validation_rules>
            - Verify all measures exist in the cube
            - Check measure compatibility
            - Ensure aggregations are appropriate
            - Validate calculation methods
            </validation_rules>

            Response format:
            '''json
            {{
                "measure_group_names": ["group1", "group2"],
                "measure_names": ["measure1", "measure2"],
                "reasoning": ["reason for selecting each measure"]
            }}
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
    """Generate OLAP query with feedback consideration"""
    try:
        feedback_section = ""
        if "user_feedback" in prev_conv:
            feedback_section = f"""
            User Feedback Analysis:
            - Previous Query: {prev_conv['feedback_query']}
            - User's Feedback: {prev_conv['user_feedback']}

            <feedback_context>
            - Consider specific issues mentioned in feedback
            - Look for syntax or structure preferences
            - Check for formatting requirements
            - Analyze if query complexity needs adjustment
            - Consider performance implications
            </feedback_context>
            """

        final_prompt = f"""
        Generate a precise OLAP cube query based on the following inputs and requirements.

        Current Context:
        - User Query: {query}
        - Selected Dimensions: {dimensions}
        - Selected Measures: {measures}

        Previous Context:
        - Previous Query: {prev_conv["query"]}
        - Previous Dimensions: {prev_conv["dimensions"]}
        - Previous Measures: {prev_conv["measures"]}
        - Previous Cube Query: {prev_conv["response"]}

        {feedback_section}

        <query_requirements>
        1. Generate a single-line OLAP query without line breaks
        2. Start with 'select' followed by dimensions then measures
        3. Use exact cube name: [Cube].[Credit One View]
        4. Include "as" aliases for all columns in double quotes
        5. Use proper dimension hierarchy references
        6. Include appropriate measure calculations
        7. Add filtering conditions if needed
        8. Ensure correct date/time handling
        </query_requirements>

        <validation_rules>
        - Verify all dimension references are valid
        - Check measure calculations are correct
        - Ensure proper syntax for filters
        - Validate temporal references
        - Check alias naming consistency
        </validation_rules>

        Examples of specific functions:
        - TimeBetween for date ranges
        - TrendNumber for year-over-year/period comparisons
        - Head/Tail for top/bottom N queries
        - RunningSum/PercentageOfRunningSum for cumulative calculations
        """
        
        result = self.llm.invoke(final_prompt)
        output = result.content
        token_details = result.response_metadata['token_usage']
        pred_query = self.cleanup_gen_query(output)
        
        return output

    except Exception as e:
        logging.error(f"Error generating query: {e}")
        raise

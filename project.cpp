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
# Add to your FastAPI file (olap_details_generat.py)

from typing import Literal, Optional
from pydantic import BaseModel

# Pydantic models for feedback
class UserFeedbackRequest(BaseModel):
    user_feedback: str
    feedback: Literal["accepted", "rejected"]
    cube_query: str
    cube_id: str

class UserFeedbackResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None

async def process_user_feedback(feedback_request: UserFeedbackRequest, user_id: str) -> dict:
    try:
        if feedback_request.feedback == "rejected":
            # Get processor instance
            if user_id not in olap_processors:
                olap_processors[user_id] = OLAPQueryProcessor(config_file)
            
            processor = olap_processors[user_id]
            history_manager = History()
            prev_conversation = history_manager.retrieve(user_id)
            
            # Process query and get new result
            query, final_query, _, dimensions, measures = processor.process_query(
                feedback_request.user_feedback,
                feedback_request.cube_id,
                prev_conversation
            )
            
            # Update conversation history
            history_manager.update(user_id, {
                "query": query,
                "dimensions": dimensions,
                "measures": measures,
                "response": final_query
            })
            
            return {
                "message": "success",
                "cube_query": final_query
            }
            
        return {
            "message": "success",
            "cube_query": feedback_request.cube_query
        }
        
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return {
            "message": "failure",
            "cube_query": None
        }

@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(
    request: UserFeedbackRequest,
    user_details: str = Depends(verify_token)
):
    user_id = f"user_{user_details}"
    result = await process_user_feedback(request, user_id)
    
    return UserFeedbackResponse(
        message=result["message"],
        cube_query=result["cube_query"]
    )

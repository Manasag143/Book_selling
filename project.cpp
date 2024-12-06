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
{
    "llm": {

        "OPENAI_API_TYPE":"azure",
        "OPENAI_API_KEY" :"c3ecb5d7c1fb4244bfb5483ff308b2f1",
        "AZURE_OPENAI_ENDPOINT" :"https://crisil-gen-ai-uat.openai.azure.com/" ,
        "OPENAI_API_VERSION" :"2024-02-15-preview",
        "DEPLOYMENT_NAME" :"gpt-4o-mini",
        "temperature": 0,
        "seed":42,
        "model":"gpt-4o-mini",
        "ENDPOINT": "https://crisil-gen-ai-uat.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
        "top_p": 0.95,
        "max_tokens": 80000
    },
    "embedding": {

        "deployment":"text-embedding-3-small",
        "show_progress_bar": "False"
    },

    "vector_embedding_path": {

        "dimensions": "./vectordb_nov24/dimensions",
        "measures": "./vectordb_nov24/measures"
    },
    "directories": {

        "excel_file_path": "./golden_30_with_gt.xlsx",
        "output_file_path": "./output/v2/gpt-4o-mini.csv"

    }

}

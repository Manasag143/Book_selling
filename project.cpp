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

FINAL_QUERY_PROMPT = '''
As an expert in generating SQL Cube queries, generate a precise single-line query based on the provided elements.

<CONTEXT>
Current Query Details:
***
User Query: {query}
Selected Dimensions: {dimensions}
Selected Measures: {measures}
Cube Name: {cube_name}
***

Previous Query Details:
***
Previous Query: {prev_query}
Previous Response: {prev_response}
***
</CONTEXT>

<QUERY_RULES>
1. Basic Structure Rules:
   - Generate single-line query without breaks
   - Include "as" aliases in double quotes
   - Use proper dimension/measure syntax
   - Include all selected dimensions and measures

2. Formatting Requirements:
   - Dimensions: [Group].[Level] as "Level"
   - Measures: [Group].[Measure] as "Measure"
   - Conditions in WHERE must use proper operators
   - String values in single quotes
</QUERY_RULES>

<FUNCTION_USAGE>
Available Functions:
+++
1. TimeBetween: For date ranges
   Example: TimeBetween(20200101,20231219,[Time].[Year], false)

2. TrendNumber: For comparisons
   Example: TrendNumber([Measures].[Value],[Time].[Year],1,'percentage')

3. Head/Tail: For top/bottom N
   Example: Head([Dimension].[Level],[Measures].[Value],5,undefined)

4. RunningSum: For cumulative totals
   Example: RunningSum([Measures].[Value])

5. FilterKPI: For conditional aggregation
   Example: FilterKPI([Measures].[Value], condition)
+++
</FUNCTION_USAGE>

<EXAMPLES>
Example Patterns:
+++
1. Basic Query:
   select [Dimension].[Level] as "Level", [Measures].[Value] as "Value" from [Cube].[{cube_name}]

2. Filtered Query:
   select [Dimension].[Level] as "Level", [Measures].[Value] as "Value" from [Cube].[{cube_name}] where [Condition] > value

3. Trend Query:
   select [Time].[Level] as "Level", [Measures].[Value] as "Value", TrendNumber([Measures].[Value],[Time].[Level],1,'percentage') from [Cube].[{cube_name}]
+++
</EXAMPLES>

<VALIDATION>
Verify the following:
1. All dimensions and measures are included
2. Proper syntax for functions
3. Correct aliasing
4. Single-line format
5. Proper cube name reference
</VALIDATION>

Generate a precise single-line cube query following these specifications.
'''
  

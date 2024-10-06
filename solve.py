import json

def load_solutions():
    """Load the solve.json file and return the data as a dictionary."""
    with open('solve.json', 'r') as file:
        return json.load(file)

def get_solution_for_issue(issue_key):
    """Fetch the solution for a specific emission issue from solve.json."""
    solutions_data = load_solutions()
    
    if issue_key in solutions_data:
        issue = solutions_data[issue_key]
        return issue['description'], issue['solutions']
    else:
        return None, None

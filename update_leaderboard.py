import pandas as pd
import os

def update_csv(user_name, accuracy, f1_score):
    file_path = 'leaderboard.csv'
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['User', 'Accuracy', 'F1 Score'])
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    
    # Add new entry
    new_data = pd.DataFrame([[user_name, accuracy, f1_score]], 
                            columns=['User', 'Accuracy', 'F1 Score'])
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Sort and save
    df = df.sort_values(by='F1 Score', ascending=False)
    df.to_csv(file_path, index=False)

# In a real scenario, you'd pull these values from the PR's output file
if __name__ == "__main__":
    # Example: update_csv("ContributorName", 0.85, 0.75)
    pass

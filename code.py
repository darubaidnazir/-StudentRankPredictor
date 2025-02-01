import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Mock Data Setup
def load_mock_data():
    # Sample NEET previous year data (Score, Rank)
    neet_data = {
        'Score': [720, 700, 680, 650, 600, 550, 500],
        'Rank': [1, 2, 3, 4, 5, 6, 7]
    }
    neet_df = pd.DataFrame(neet_data).sort_values('Score', ascending=False)
    
    # Sample College cutoffs
    colleges = {
        'College': ['AIIMS Delhi', 'CMC Vellore', 'JIPMER Puducherry'],
        'Cutoff_Rank': [1, 5, 7]
    }
    college_df = pd.DataFrame(colleges)
    
    # Mock Quiz Data for a user
    questions = [
        {'id': 1, 'subject': 'Biology', 'difficulty': 'Easy', 'correct_option': 1},
        {'id': 2, 'subject': 'Biology', 'difficulty': 'Medium', 'correct_option': 3},
        {'id': 3, 'subject': 'Chemistry', 'difficulty': 'Medium', 'correct_option': 2},
        {'id': 4, 'subject': 'Physics', 'difficulty': 'Hard', 'correct_option': 4},
    ]
    
    submissions = [
        {'question_id': 1, 'selected_option': 1},
        {'question_id': 2, 'selected_option': 3},
        {'question_id': 3, 'selected_option': 2},
        {'question_id': 4, 'selected_option': 4},
    ]
    
    historical = [
        {'scores': [70, 75, 80, 85, 90], 'subjects': ['Biology']*5},
    ]
    
    return neet_df, college_df, questions, submissions, historical

# Load Data
neet_df, college_df, questions, submissions, historical = load_mock_data()

# Process Quiz Data
def process_submissions(questions, submissions):
    data = []
    for q in questions:
        sub = next(s for s in submissions if s['question_id'] == q['id'])
        correct = 1 if sub['selected_option'] == q['correct_option'] else 0
        data.append({
            'subject': q['subject'],
            'difficulty': q['difficulty'],
            'correct': correct
        })
    return pd.DataFrame(data)

current_df = process_submissions(questions, submissions)

# Analyze Performance
def analyze_performance(df):
    # Accuracy by Subject and Difficulty
    accuracy = df.groupby(['subject', 'difficulty'])['correct'].agg(['sum', 'count'])
    accuracy['accuracy'] = accuracy['sum'] / accuracy['count']
    return accuracy.reset_index()

performance = analyze_performance(current_df)

# Predict NEET Score
neet_structure = {
    'Biology': {'Easy': 30, 'Medium': 50, 'Hard': 10},
    'Chemistry': {'Easy': 20, 'Medium': 50, 'Hard': 20},
    'Physics': {'Easy': 10, 'Medium': 40, 'Hard': 40}
}

def predict_neet_score(performance_df, neet_structure):
    total_correct = 0
    total_incorrect = 0
    
    for subject in neet_structure:
        for difficulty in neet_structure[subject]:
            count = neet_structure[subject][difficulty]
            filtered = performance_df[
                (performance_df['subject'] == subject) & 
                (performance_df['difficulty'] == difficulty)
            ]
            if not filtered.empty:
                acc = filtered.iloc[0]['accuracy']
            else:
                acc = 0  # Default if no data
            correct = acc * count
            incorrect = (1 - acc) * count
            total_correct += correct
            total_incorrect += incorrect
    
    score = (total_correct * 4) - (total_incorrect * 1)
    return score

predicted_score = predict_neet_score(performance, neet_structure)

# Predict Rank
def predict_rank(score, neet_df):
    neet_df = neet_df.sort_values('Score', ascending=False)
    neet_df['cum_rank'] = np.arange(1, len(neet_df) + 1)
    for idx, row in neet_df.iterrows():
        if score >= row['Score']:
            return row['Rank']
    return len(neet_df) + 1  # If score lower than all

predicted_rank = predict_rank(predicted_score, neet_df)

# Predict College
def predict_college(rank, college_df):
    eligible = college_df[college_df['Cutoff_Rank'] >= rank]
    return eligible['College'].tolist()

predicted_colleges = predict_college(predicted_rank, college_df)

# Generate Insights
def generate_insights(performance_df, historical):
    # Weak Areas
    weak_areas = performance_df.groupby('subject')['accuracy'].mean().idxmin()
    
    # Improvement Trends
    trends = {}
    scores = historical[0]['scores']
    X = np.array(range(len(scores))).reshape(-1, 1)
    model = LinearRegression().fit(X, scores)
    slope = model.coef_[0]
    trends['Overall'] = 'Improving' if slope > 0 else 'Declining' if slope < 0 else 'Stable'
    
    return {'weak_areas': weak_areas, 'trends': trends}

insights = generate_insights(performance, historical)

# Visualization
plt.figure(figsize=(10, 4))
performance.groupby('subject')['accuracy'].mean().plot(kind='bar')
plt.title('Average Accuracy by Subject')
plt.ylabel('Accuracy')
plt.savefig('accuracy_by_subject.png')

# Output Results
print(f"Predicted NEET Score: {predicted_score}")
print(f"Predicted Rank: {predicted_rank}")
print(f"Eligible Colleges: {predicted_colleges}")
print(f"Insights: Weak Area - {insights['weak_areas']}, Trend - {insights['trends']['Overall']}")

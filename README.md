To address the Student Rank Predictor problem, I developed a solution using Python that analyzes quiz performance data to predict NEET ranks and potential college admissions. Here's a step-by-step explanation:

Approach
Data Collection & Processing:

Mock Data: Created mock datasets for quiz submissions and historical NEET results due to unavailable APIs.

Aggregation: Compiled user responses across quizzes, categorizing by subject (Biology, Chemistry, Physics) and difficulty (Easy, Medium, Hard).

Performance Analysis:

Accuracy Calculation: Determined accuracy per subject and difficulty.

Trend Detection: Applied linear regression to historical quiz scores to identify improvement trends.

Score Prediction:

NEET Score Estimation: Used accuracy metrics and NEET's marking scheme (correct: +4, incorrect: -1) to predict total scores.

Rank Prediction:

Historical Data Mapping: Compared predicted scores against previous year's NEET data to estimate ranks.

College Prediction:

Cutoff Ranks: Utilized mock college cutoff data to determine likely admissions based on predicted ranks.

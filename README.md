Lung Cancer Detection using Machine Learning
📄 Project Overview
This repository contains the code and analysis for a machine learning project focused on predicting Lung Cancer risk based on behavioral and symptomatic data. The primary goal is to build an effective classification model that can assist in preliminary risk assessment.

The project involves comprehensive Exploratory Data Analysis (EDA), data preprocessing (including handling categorical features and scaling), and training various supervised classification models to identify the most predictive features and achieve high accuracy.

💾 Dataset
The analysis utilizes the survey lung cancer.csv dataset, which contains 309 entries and 16 features.

Key Features:
The features are primarily binary indicators (1 or 2) representing the presence or absence of a symptom or behavior, along with demographic data.

Column Name

Description

Data Type

GENDER

Male or Female

Object (Converted to Numerical)

AGE

Age of the subject

Integer

SMOKING

Smoking status

Integer (1/2)

YELLOW FINGERS

Presence of yellow fingers

Integer (1/2)

ANXIETY

Level of anxiety

Integer (1/2)

CHRONIC DISEASE

Presence of a chronic disease

Integer (1/2)

FATIGUE

Level of fatigue

Integer (1/2)

...

and 8 more symptomatic features

Integer (1/2)

LUNG CANCER

Target variable (YES or NO)

Object (Converted to Numerical)

🛠️ Methodology and Analysis
1. Data Preprocessing
Label Encoding: Categorical features (GENDER, LUNG CANCER) were converted into numerical representations (0 and 1) using LabelEncoder.

Train-Test Split: The data was split into training and testing sets (e.g., a 80/20 split) to ensure model performance evaluation on unseen data.

Feature Scaling: Numerical features were scaled using StandardScaler to normalize the range of independent variables.

2. Exploratory Data Analysis (EDA)
The analysis included:

Visualizing the distribution of the target variable (LUNG_CANCER) using count plots, indicating a class imbalance.

Analyzing the distribution of AGE and its relationship with the cancer outcome using histograms and box plots.

Generating a Correlation Heatmap to understand the relationships between all numerical features.

3. Model Training
Two primary classification algorithms were trained:

Logistic Regression

Support Vector Classifier (SVC)

📊 Results
Both models demonstrated high performance metrics on the test set.

SVC Performance Summary
The Support Vector Classifier achieved the following score:

Test Accuracy: ≈96.77%

Training Accuracy: ≈85.02%

Logistic Regression Metrics
Metric

Score

Accuracy

0.9677

Precision

0.9833

Recall

0.9833

F1-score

0.9833

(Note: While overall accuracy is very high, the confusion matrix and classification report highlighted challenges in correctly predicting the minority class (No Cancer), suggesting the model is heavily biased toward the majority class due to class imbalance.)

💻 Technologies Used
Python 3.x

Pandas (Data manipulation and analysis)

NumPy (Numerical operations)

Scikit-learn (Machine Learning models and utilities: LogisticRegression, SVC, StandardScaler, train_test_split)

Matplotlib (Data visualization)

Seaborn (Advanced data visualization)

🚀 Getting Started
To run this project locally, you will need a Python environment with the required libraries installed.

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn

Run the analysis:
Execute the code contained in the main notebook or script (e.g., lung_cancer_detection.ipynb).

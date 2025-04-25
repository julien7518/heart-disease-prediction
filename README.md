# Heart Disease Prediction - Model evaluation and comparison

This assignment, part of the Data Analysis course at Dorset College, focuses on predicting heart disease using a synthetic dataset. The goal is to explore the data, apply preprocessing techniques, and assess the performance of different classification algorithms.

## 1. Data Exploration and Preprocessing

The first step of the project was to take a closer look at the dataset and get it ready for training machine learning models. This includes checking the structure of the data, looking for missing values, and making sure everything is in the right format.

### Data summary

After checking the dataset, we saw that it was already fully preprocessed. The features were normalized and standardized, which means they were all on the same scale. There were no missing values, and all columns were in numerical format: floats for the features and an integer for the target column. Since no additional data cleaning or transformation was needed, the dataset will be used as it is in the next step.

> [!NOTE]
> All results obtained from the EDA are available in the [appendix](#appendix).

### Visualisations

A first heatmap was made to show the correlations between all the numerical features in the dataset. This is something we usually do in data analysis to get an overview of how the variables are related to each other, including the target.  
But since in this project we’re mainly interested in predicting the target (last column of the matrix), we also created a second heatmap that only shows the correlation between each feature and the target. This gives a clearer idea of which features are the most important for our predictions.
<div style="display: flex; gap: 20px; align-items: center; margin-bottom: 40px;justify-content: center;">
  <img src="resources/heatmap.png" alt="Correlation heatmap" style="height: 400px;">
  <img src="resources/target_heatmap.png" alt="Target correlation heatmap" style="height: 400px;">
</div>

We also made a chart to show how the target classes are distributed. This helps us see if the dataset is balanced, which is important to judge how well the models really perform.

<div style="display: flex; align-items: center; margin-bottom: 40px; justify-content: center;">
  <img src="resources/distributions.png" alt="Correlation heatmap" style="height: 400px;">
</div>

## 2. Model Training



## 3. Model Evaluation



## 4. Comparative Analysis



## Reflection Questions



## Appendix

### Result of the EDA

| Feature           | Count | Mean      | Std      | Min        | 25%       | 50%       | 75%       | Max      |
|:------------------|:-----:|:---------:|:--------:|:----------:|:---------:|:---------:|:---------:|:--------:|
| age               | 500.0 | -0.240697 | 2.236264 | -6.637937  | -1.731686 | -0.271554 | 1.418476  | 5.091704 |
| resting_bp        | 500.0 | 0.002554  | 1.011668 | -2.891833  | -0.650030 | 0.032846  | 0.707191  | 2.748645 |
| cholesterol       | 500.0 | 0.952978  | 2.027121 | -5.591194  | -0.344919 | 1.088140  | 2.298920  | 6.563156 |
| max_hr            | 500.0 | 0.891863  | 1.990966 | -5.979594  | -0.235558 | 1.035698  | 2.226467  | 6.346779 |
| oldpeak           | 500.0 | 0.599015  | 2.858443 | -6.915749  | -1.563770 | 0.327706  | 3.141920  | 6.440231 |
| num_major_vessels | 500.0 | -0.126711 | 2.363934 | -6.749837  | -1.835318 | -0.183129 | 1.423795  | 7.014193 |
| sex               | 500.0 | -1.617820 | 1.592002 | -7.218200  | -2.622877 | -1.671415 | -0.686045 | 3.557356 |
| chest_pain_type   | 500.0 | 0.089429  | 2.126706 | -4.992435  | -1.465407 | -0.077492 | 1.794532  | 6.771795 |
| fasting_bs        | 500.0 | 0.906553  | 1.882720 | -7.019296  | -0.243922 | 1.066285  | 2.206591  | 5.803134 |
| exercise_angina   | 500.0 | -2.629439 | 2.956567 | -12.484286 | -4.719322 | -2.454379 | -0.062064 | 4.434569 |
| target            | 500.0 | 0.408000  | 0.491955 | 0.000000   | 0.000000  | 0.000000  | 1.000000  | 1.000000 |

```
+---------------+
| Dataset infos |
+---------------+
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500 entries, 0 to 499
Data columns (total 11 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   age                500 non-null    float64
 1   resting_bp         500 non-null    float64
 2   cholesterol        500 non-null    float64
 3   max_hr             500 non-null    float64
 4   oldpeak            500 non-null    float64
 5   num_major_vessels  500 non-null    float64
 6   sex                500 non-null    float64
 7   chest_pain_type    500 non-null    float64
 8   fasting_bs         500 non-null    float64
 9   exercise_angina    500 non-null    float64
 10  target             500 non-null    int64  
dtypes: float64(10), int64(1)
memory usage: 43.1 KB
None
+----------------+
| Missing values |
+----------------+
age                  0
resting_bp           0
cholesterol          0
max_hr               0
oldpeak              0
num_major_vessels    0
sex                  0
chest_pain_type      0
fasting_bs           0
exercise_angina      0
target               0
dtype: int64
```
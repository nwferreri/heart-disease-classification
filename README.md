# Heart Disease Classification

This project uses Python-based machine learning and data science libraries in an attempt to build a machine learning model that can predict if someone has heart disease.

## Problem

Given clinical parameters about a patient, can we predict whether or not they have heart disease?

## Data

The original data came from the Cleveland database from the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease

There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

It has ~300 entries with the following features:
1. age - age in years
2. sex - (1 = male; 0 = female)
3. cp - chest pain type
    * 0: Typical angina: chest pain related decrease blood supply to the heart
    * 1: Atypical angina: chest pain not related to heart
    * 2: Non-anginal pain: typically esophageal spasms (non heart related)
    * 3: Asymptomatic: chest pain not showing signs of disease
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
5. chol - serum cholesterol in mg/dL
    * serum = LDL + HDL + .2 * triglycerides
    * above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dL) (1 = true; 0 = false)
    * '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
    * 0: Nothing to note
    * 1: ST-T Wave abnormality
        * can range from mild symptoms to severe problems
        * signals non-normal heart beat
    * 2: Possible or definite left ventricular hypertrophy
        * Enlarged heart's main pumping chamber
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest, looks at stress of heart during exercise, unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
    * 0: Upsloping: better heart rate with exercise (uncommon)
    * 1: Flatsloping: minimal change (typical healthy heart)
    * 2: Downsloping: signs of unhealthy heart
12. ca - number of major vessels (0-3) colored by fluoroscopy
    * colored vessel means the doctor can see the blood passing through
    * the more blood movement the better (no clots)
13. thal - thallium stress result
    * 1,3: normal
    * 6: fixed defect: used to be defect but ok now
    * 7: reversable defect: no proper blood movement when exercising
15. target - have disease or not (1 = yes, 0 = no) (= the predicted attribute)

## Evaluation Metric

If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

## Exploratory Data Analysis

To start, EDA was performed to gain more familiarity with the data.

First, we looked at the balance of the dataset. There are 165 instances with heart disease and 138 without, indicating a fairly well-balanced dataset for the target variable.

Next, we explored the frequency of heart disease according to sex, seen in **Figure 1** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/42e49a9a-c84b-46a6-896c-5ce9f1abb059)<br>
**Figure 1**: The data contains more entries for men than women, leading to higher instances of both cases for men. It also shows that women seem more likely to be diagnosed with heart disease.

Next, we explored how age and max heart rate are related for those with and without heart disease, seen in **Figure 2** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/cbd07f21-4d13-43ad-beef-5025eb777747)<br>
**Figure 2**: Overall, heart rate decreases with age in both groups. There also appears to be more variability in max heart rate for those without heart disease.

A histogram of ages in the dataset is shown in **Figure 3** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/26e63976-be30-4b28-a84d-0df66beca66c)<br>
**Figure 3**: Fairly normal distribution of ages, with a slight skew toward older individuals, which makes sense when thinking about those at higher risk for heart disease.

Next, we explored heart disease frequency as a function of the different types of chest pain, seen in **Figure 4** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/b43e491d-513d-4489-a2b7-c634743a4d78)<br>
**Figure 4**: The highest incidence of heart disease is in the group with non-anginal (non heart related) chest pain, which is not what we expected. In addition, the highest incidence of no heart disease was linked with typical angina, or chest pain related to a decrease in blood supply to the heart. Again, a counterintuitive result.

In **Figure 5** below, a correlation matrix shows the relationship between all the variables in the dataset.<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/943bb976-5f5b-46fa-8a51-bfe476aa2b17)<br>
**Figure 5**: Correlation matrix shows several relatively high positive and negative correlations in the data, especially among the target variable and others.

For a final exploration, we looked at age and resting blood pressure as a function of heart disease occurrence, seen in **Figure 6** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/258d74b1-bf43-4a03-8261-0270140ed1ac)<br>
**Figure 6**: There seems to be a slight increase in resting blood pressure after age 50 for both groups, but no easily discernable difference between the groups.

## Modeling

Three different machine learning models were trained on the data and their baseline accuracy was assessed. Results can be seen in **Figure 7** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/80859632-64cd-4eb9-8160-036d571ddd69)<br>
**Figure 7**: Logistic Regression performed best, closely followed by Random Forest. K-Nearest Neighbors performed significantly worse than the others.

## Hyperparameter Tuning

Hyperparameter tuning (with cross-validation, when possible) was performed in an attempt to improve the accuracy of all three models.

### K-Nearest Neighbors
We tested the KNN model across 20 different values for the `neighbors` parameter, with the results graphed in **Figure 8** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/cd0a0a15-1a42-433f-bc86-046bf3d4180d)<br>
**Figure 8**: The best accuracy score on the test data was ~75% using 8 neighbors, which is an improvement over the initial model, but still worse than the other two models. Based on this, we will drop KNN from further testing.

### Random Forest

The first tuning pass for the Random Forest model was a randomized search of parameter values, which yielded an increase in accuracy from 83.6% to 86.9%. While better, this is still lower than the baseline accuracy of the Logistic Regression, so we will drop the Random Forest from further testing.

### Logistic Regression

The first tuning pass for the Logistic Regression model was a randomized search of parameter values, which did not improve the initial accuracy of 88.5%.

The second tuning pass was a grid search of parameter values, which again, did not improve the accuracy of the model. While this does not reach the target evaluation accuracy of 95%, it is high enough to warrant exploring the model's performance across other metrics.

## Model Evaluation

First, we created a ROC curve and calculated the AUC score, seen in **Figure 9** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/d3e5272b-5590-42c8-8da6-97784ef32cf7)<br>
**Figure 9**: This shows that the model does a very good job of correctly classifying patients with heart disease, while also not generating a large number of false positives.

To further explore where the model is struggling, a confusion matrix was created, seen in **Figure 10** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/a42489af-280a-4757-8d79-f852caec884e)<br>
**Figure 10**: High incidence of correct classifications and low incidence of incorrect classifications.

Next, we calculated cross-validated classification metrics, visualized in **Figure 11** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/d11ee8b9-5a57-4c40-b1d2-729f9fcc1719)<br>
**Figure 11**: Cross-validated accuracy, precision, recall, and F1 scores.

Finally, the feature importances for the Logistic Regression model were extracted and visualized in **Figure 12** below:<br>
![image](https://github.com/nwferreri/heart-disease-classification/assets/112211174/1e776f62-babb-436f-8d72-7854f1269583)<br>
**Figure 12**: The features with the most positive contributions are `cp` (chest pain type), `slope` (heart rate change during exercise), and `restecg` (resting electrocardiograph results). The features with the most negative contributions are `sex`, `thal` (thallium stress result), and `ca` (number of major vessels colored by fluoroscopy).

## Evaluation

The best model we could generate was a Logistic Regression model with 88.5% accuracy on the test data. This does not meet the evaluation criteria defined at the outset of the project, though it is close.

Further improvements may be possible by gathering more data or using another machine learning model.

The best model was exported for future reference.

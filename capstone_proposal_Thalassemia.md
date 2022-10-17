# Capstone Proposal

Guillermo Tarrasó Urios

30/09/2022


## Domain Background

Thalassemia is an inherited (i.e., passed from parents to children through genes) blood disorder caused when the body doesn’t make enough of a protein called hemoglobin, an important part of red blood cells. When there isn’t enough hemoglobin, the body’s red blood cells don’t function properly and they last shorter periods of time, so there are fewer healthy red blood cells traveling in the bloodstream.

Red blood cells carry oxygen to all the cells of the body. Oxygen is a sort of food that cells use to function. When there are not enough healthy red blood cells, there is also not enough oxygen delivered to all the other cells of the body, which may cause a person to feel tired, weak or short of breath. This is a condition called anemia. People with thalassemia may have mild or severe anemia. Severe anemia can damage organs and lead to death.

https://www.cdc.gov/ncbddd/thalassemia/facts.html

## Problem Statement

An effective screening program to detect Thalassemia carriers is vital to Thalassemia prevention. There are many challenges to an effective screening program, especially in low-resource settings. Considering alpha-thalassemia, genetic testing is needed for a confirmatory diagnosis of a carrier, which is expensive and not widely available. Machine learning (ML) models can act as decision-support tools that are easy to deploy and use in low-resource settings.

## Datasets and Inputs

This dataset (https://www.kaggle.com/datasets/letslive/alpha-thalassemia-dataset) is from a database of 288 cases from the Human Genetics Unit (HGU) of the Faculty of Medicine, Colombo, Sri Lanka. It's collected from Alpha thalassemia carrier children and their family members screened from 2016 to 2020.

There are 15 independent variables and apart from Sex ('male' and 'female') all others are continuous variables in float format (identifier in brackets):

•	Sex
•	Hemoglobin concentration in grams per decilitre - g/dL (hb)
•	Pack cell volume/hematocrit in % (pcv)
•	Red blood cell volume in 10^12/L (rbc)
•	Mean cell volume in femtolitres-fl (mcv)
•	Mean corpuscular hemoglobin in picograms-pg (mch)
•	Mean corpuscular hemoglobin concentration in grams per decilitre-g/dL (mchc)
•	Red blood cell distribution width in % (rdw)
•	Total white blood cell count in 10^6/L (wbc) with white blood cell types in % - (neut, lymph)
•	Total platelet count in 10^6/L (plt)
•	Hemobglobin A, A2 and F in % from HPLC testing (hba, hba2, hbf)

The target variable is phenotype, this is different in the two csv files:

•	in alphanorm.csv: alpha carrier and normal (binary categorical)
•	in twoalphas.csv : alpha trait and silent carrier (binary categorical)


## Solution Statement

In the aim of build a ML model able to correctly predict the phenotype of the patients, different kind of classification algorithms will be tested: ExtraTreesClassifier, XGBClassifier, RandomForestClassifier, AdaBoostClassifier. With the model with best performance a specific search of the best hyperparameters will made with validation curves. This process will be done in parallel in both datasets, building two different models.

## Benchmark Model

Scikit-Learn’s Dummy Classifier will be used as the benchmark model. As the majority of samples are normal (alphanorm.csv) or silent carrier (twoalphas.csv), the dummy model will be set in stratified mode. Thus, we will see how good is the model to detect specifically the alphas carriers and alpha trait respectively comparatively with the dummy model.

## Evaluation Metrics

Both Dummy classifier and the selected classifier models will be evaluated with the metrics recall, f1 and fbeta. In this kind of test, to detect all the not normal phenotypes is the main goal. It means as less false negatives the better, even some false positives could happen. This is why have chosen recall as first evaluation metrics, f1 to see the global performance between recall and precision, and finally fbeta to see a whole vision but with recall predominance.


## Project Design


1. Once the dataset is loaded, I will look through the data and perform exploratory data analysis. To ensure all samples and features are in the way we need it, one hot encoding, fill empty cells, drop columns and change columns datatype will performed if is needed.
2. Different algorithms will tested in a first round, to select the best performance. 
3. I will split the data in training and testing sets.
4. Using the best algorithm in first round, looking for the best hyperparameters of it with Cross Validation
5. Evaluate the model's performance on testing data using the above evaluation metrics, comparing it with the Dummy model
6. Analyse results.



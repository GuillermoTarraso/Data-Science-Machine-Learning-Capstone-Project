# Capstone Report 


## Abstract


Thalassemia is an inherited disease that requires an adequate diagnosis for the treatment and following. The diagnostic test is not always easy to obtain in terms of speed or it is not affordable for the patient.
The gold standard for diagnosis is genetic testing. This kind of test may have high cost. Depending the patient's country or region, the test could not be available due to the equipment requirements, or if it is available it takes a long time to be done.
In Thalassemia’s patients the blood cells will be affected, and this alterations are the base for suspicious and they oriented the clinicians to the diagnostic. The aim of this project is determinate if a Machine Learning algorithm can predict the genetic phenotype of the patients based on the blood parameters, less complex to obtain, cheaper and faster.
Classification algorithms could solve the problem, but in all diseases exist an interindividual variability, and maybe the different phenotypes overlap at some parameters, and the classification become harder. Besides other clinical symptoms and signs of the thalassemia that play an important role in the diagnostic maybe are not included in the dataset. 
The models developed has minor improvement in recall, f1 and fbeta against the dummy model in the comparison normal vs carrier. When we evaluate this metrics comparing alpha trait vs silent carrier, the model shows a significant improvement, but still far from be a good alternative to genetic testing 


## Introduction

Thalassemia is an inherited (i.e., passed from parents to children through genes) blood disorder caused when the body doesn’t make enough of a protein called hemoglobin, an important part of red blood cells. When there isn’t enough hemoglobin, the body’s red blood cells don’t function properly and they last shorter periods of time, so there are fewer healthy red blood cells traveling in the bloodstream.

Red blood cells carry oxygen to all the cells of the body. Oxygen is a sort of food that cells use to function. When there are not enough healthy red blood cells, there is also not enough oxygen delivered to all the other cells of the body, which may cause a person to feel tired, weak or short of breath. This is a condition called anemia. People with thalassemia may have mild or severe anemia. Severe anemia can damage organs and lead to death. https://www.cdc.gov/ncbddd/thalassemia/facts.html

Thalassemia requires an adequate diagnosis for the treatment and following. The diagnostic test is not always easy to obtain in terms of speed or it is not afordable for the patient. The gold standard for diagnosis is genetic testing. This kind of test may have high cost. Depending the patient's country or region, the test could not be available due to the equipment requirements, or if it is available it takes a long time to be done.

In Thalassemia’s patients, the lack oh hemoglobin and low level of oxygen in blood causes the alterations of blood parameters which can be detected in a hemogram. This kind of test is done routinely in hospitals laboratories, and the alteration of blood parameters are the base for suspicious and they oriented the clinicians to the diagnostic. 

In this work I have used the dataset (https://www.kaggle.com/datasets/letslive/alpha-thalassemia-dataset) from a database of 288 cases from the Human Genetics Unit (HGU) of the Faculty of Medicine, Colombo, Sri Lanka. It's collected from Alpha thalassemia carrier children and their family members screened from 2016 to 2020. It has 11 blood parameters in form of continuous variables, besides the sex and phenotype. This dataset has two different databases inside, one with alpha carriers and healthy people and other with alpha traits and silent carriers

With this dataset I have created a machine learning algorithm to predict the phenotype of each person in the list based only in blood parameters, avoiding the genetic test for the diagnostic. This algorithm could be used in the situation when genetic test is impossible, no access to a specialized laboratory or unaffordable cost of the test.


## Related Work

The TVGH-NYCU Thal research group in Taiwan had faced to a similar problem recently. They have created a ML Classifier to differentiate the alpha Thalassemia patients and beta thalassemia patients from other blood disorders, in this case Iron deficiency. Iron deficiency patients has similar clinic signs that thalassemia patients, so differential diagnosis is very important. They has adopted support vector machine (SVM) with Monte-Carlo cross-validation procedure to generate the classifier. The performance of their classifier was compared with original indices by calculating the average classification error rate and area under the curve (AUC) for the sampled datasets. 

## Methodology

### Dataset

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

I have chosen this dataset because as a health professional working in al lab it seems very interesting to me. Predicting a phenotype based on blood parameters in a disease that causes alterations in blood parameters has perfect sense to me. Besides, this is a good dataset for ML since it has mostly continuous variables. Despite you can use one hot encoding, in my short experience you will obtain better performance if you can use the data as it is, maybe it's just a feeling. In the same idea of changing the data as lees as possible, there are only few (2-3) empty cell in each dataset, which I fill with the mean, and as I'm using classification algorithms I didn't need to scale the data. The other small modification that the datasets require is to convert the sex feature in numerical (0/1) and we are ready to use both datasets.


### Baseline Model

I don't know other models used previously in this precise dataset, so I choose the Scikit-Learn’s Dummy Classifier model as baseline model. One of the most important aspects of the model is the recall. As the majority of samples are normal (alphanorm.csv) or silent carrier (twoalphas.csv), the dummy model in prior strategy will cause 0 recall, easy to beat, so I set it in stratified mode.

### Algorithms

After divide the dataset in train and test set with train_test_split, I started selecting the algorithm with best performance in terms of  recall and f1. For this I used StratifiedShuffleSplit, and the algorithms tested were ExtraTreesClassifier, XGBClassifier, RandomForestClassifier, AdaBoostClassifier and the Dummy Classifier. 

In this first approximation, the two datasets gave different results. In the first case (alpha carrier and normal phenotypes) Adaboost achieve the best scores, so I decide to use it in the next steps. 
I created validation curves to select the best hyperparameters for Adaboost, so I used validation curves with StratifiedKFold to select the best number of estimators and learning rate. Then I fit the Adaboost model with best hyperparameters and I made a learning curve also with StratifiedKFold to check for possible overfittings. I plotted the confusion matrix for models (AdaBoost and Dummy) and finally I have calculated recall f1 and fbeta. 

In the other hand, with the second dataset (alpha trait and silent carrier phenotypes) ExtraTreesClassifier get the highest scores, so I used this algorithm in the second dataset. As with the first data set, I made learning curves to select the best max_depth and min_samples_split, and with the best options I fit the model. Then, I plotted a learning curve and confusion matrix as well, to finally evaluate the model with the chosen metrics. 

At this point as the model performed well I decided to do a feature selection calculating the importance of each feature and plotting them in a graph. The lowest contributor was sex, so I created a new dataframe dropping the sex feature. After that I decide to go further with a PCA.  I need to scale the data with MinMaxScaler and split in train and test first. Plotting the explained variance of the components, I saw that two components gets most the explained variation, so I made a scatterplot expecting see two groups of dots of the two phenotypes.

![imagen](https://user-images.githubusercontent.com/115868725/196647409-0e482371-1fc5-4926-9a64-d16353941391.png)


### Metrics

Both Dummy classifier and the selected classifier models will be evaluated with the metrics recall, f1 and fbeta. In this kind of test, to detect all the not normal phenotypes is the main goal. It means as less false negatives the better, even some false positives could happen. This is why have chosen recall as first evaluation metrics, f1 to see the global performance between recall and precision, and finally fbeta to see a whole vision but with recall predominance. 

### Experiments

First I used StratifiedShuffleSplit to select the best Clasiffier to my dataset. Once selected I made validation curves for select the best hyperparameters. Then, I made learning curves to see the performance depending the number of samples and check for possible under or overfitting.

## Results and Analysis

In the first stage of screening to select the best algorithm I found only small differences in both datasets. In the carrier/normal dataset, mostly models perform around f1 = 0.5, the same as the Dummy model. Adaboost rise 0.53 so that was the reason I selected it. In the alpha trait/silent carrier dataset, things were different and the models rise 0.61 as minimum to 0.68 of ExtraTreeClassifier, comparised with 0.49 of the dummy model. After selecting the best hyperparameters and fitting both models, in the fisrt dataset the comparison with the dummy model in recall, f1 and  fbeta was: 

![imagen](https://user-images.githubusercontent.com/115868725/196647801-bfefd81d-4890-45d2-8db5-a7afd1953ecd.png)


In the second dataset the three metrics show the following results:

![imagen](https://user-images.githubusercontent.com/115868725/196648044-d3411916-844f-4e9b-9d9b-8cbd1d840284.png)

In the alpha trait/silent carrier dataset, after scaling the data and dropping the sex feature, the value of fbeta rises 0.82. 

## Conclusion

The developed models has success beating the benchmark model, the Scikit-Learn’s Dummy Classifier. Despite of that, in the dataset of carrier/normal, the improvement is too slow, I consider it is far from be a usable model in a real situations. Probably the differences between both groups are too small to build a good Classifier, and more data (adding more relevant features) is needed to improve the model. 
The second model is significantly better than the dummy model, with values in its metrics that makes it a point of start for a model to implement in clinical practice. 

Surprises me that after PCA, the 2 principal component explain 83% of the variance, but it is not reflected creating a group of dots for each phenotype. 

![imagen](https://user-images.githubusercontent.com/115868725/196647502-8ce6a3eb-e24f-4990-b344-32950f56b220.png)![imagen](https://user-images.githubusercontent.com/115868725/196647642-70b283c2-9651-462d-9d90-0781f83bdcf7.png)


Perhaps high variance at some features doesn’t imply a high significance for classification.






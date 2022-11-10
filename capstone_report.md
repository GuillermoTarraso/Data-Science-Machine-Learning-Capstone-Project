# Capstone Report 


## Abstract


Alpha thalassemia is an inherited disease that requires an adequate diagnosis for the treatment and following. The diagnostic test is not always easy to obtain in terms of speed or it is not affordable for the patient.
This project aims to determine if a Machine Learning algorithm can predict the genetic phenotype of the patients based on the blood parameters, less complex to obtain, cheaper, and faster than other procedures.
After testing and optimization differents classification algorithms, two models have been developed in this project. First classification model developed (Alphanorm, comparing normal individual vs a individual carrier of the desease) have a minor improvement in the model evaluation metrics against the dummy model. When we evaluate these metrics in the second model(Twoalphas) by comparing two kinds of carriers (silent carrier vs alpha trait), the model shows a significant improvement, but is still far from being a good alternative to genetic testing. 


## Introduction

Alpha thalassemia is an inherited (i.e., passed from parents to children through genes) blood disorder caused when the body doesn’t make enough of a protein called hemoglobin, an important part of red blood cells. When there isn’t enough hemoglobin, the body’s red blood cells don’t function properly and they last shorter periods, so fewer healthy red blood cells are traveling in the bloodstream.

Red blood cells carry oxygen to all the cells of the body. Oxygen is a sort of food that cells use to function. When there are not enough healthy red blood cells, there is also not enough oxygen delivered to all the other cells of the body, which may cause a person to feel tired, weak, or short of breath. This is a condition called anemia. People with thalassemia may have mild or severe anemia. Severe anemia can damage organs and lead to death. https://www.cdc.gov/ncbddd/thalassemia/facts.html

In Alpha thalassemia patients, the lack of hemoglobin and low level of oxygen in the blood causes alterations in blood parameters which can be detected in a hemogram. This kind of test is done routinely in hospital laboratories, and the alteration of blood parameters is the base for suspicion and they oriented the clinicians to the diagnostic. Despite of that, Alpha thalassemia requires an adequate diagnosis for the treatment and following. The diagnostic test is not always easy to obtain in terms of speed or it is not affordable for the patient. The gold standard for diagnosis is genetic testing. This kind of test may have a high cost. Depending on the patient's country or region, the test could not be available due to the equipment requirements, or if it is available it takes a long time to be done.  

Alpha thalassemia occurs when some or all of the 4 genes that make hemoglobin (the alpha-globin genes) are missing or damaged.

There are 4 types of [alpha thalassemia](https://www.hopkinsmedicine.org/health/conditions-and-diseases/alpha-thalassemia):

- Alpha thalassemia silent carrier. One gene is missing or damaged, and the other 3 are normal. Blood tests are usually normal. Your red blood cells may be smaller than normal. Being a silent carrier means you don’t have signs of the disease, but you can pass the damaged gene on to your child. This is confirmed by DNA tests.
- Alpha thalassemia carrier. Two genes are missing. You may have mild anemia.
- Hemoglobin H disease. Three genes are missing. This leaves just 1 working gene. You may have moderate to severe anemia.  You have a greater risk of having a child with alpha thalassemia major.
- Alpha thalassemia major. All 4 genes are missing. This causes severe anemia. 

In this work, I have used the [dataset](https://www.kaggle.com/datasets/letslive/alpha-thalassemia-dataset) from a database of 288 cases from the Human Genetics Unit (HGU) of the Faculty of Medicine, Colombo, Sri Lanka. It's collected from Alpha thalassemia carrier children and their family members screened from 2016 to 2020. It has 11 blood parameters in form of continuous variables, besides sex and phenotype. This dataset has two different databases inside, one with alpha carriers and healthy people and the other with alpha traits and silent carriers

With this dataset I have created two ML clasaification to predict the phenotype of each person in the list based only on blood parameters, avoiding the genetic test for the diagnostic. These models can diferenciate phenotypes, the first model between normal and carrier individuals and second between silent carriers and alpha tarit individuals. Both  models could be used in a situation when a genetic test is impossible, there is no access to a specialized laboratory, or the cost of the test is unaffordable.


## Related Work

The TVGH-NYCU Thal research group in Taiwan had faced a similar problem recently. They have created an ML Classifier to differentiate the alpha Thalassemia patients and beta Thalassemia patients from other blood disorders, in this case, Iron deficiency. Iron deficiency patients have similar clinical signs that thalassemia patients, so differential diagnosis is very important. They have adopted a support vector machine (SVM) with a Monte-Carlo cross-validation procedure to generate the classifier. The performance of their classifier was compared with the original indices by calculating the average classification error rate and area under the curve (AUC) for the sampled datasets.


## Methodology

### Dataset

[This dataset](https://www.kaggle.com/datasets/letslive/alpha-thalassemia-dataset) is from a database of 288 cases from the Human Genetics Unit (HGU) of the Faculty of Medicine, Colombo, Sri Lanka. It's collected from Alpha thalassemia carrier children and their family members screened from 2016 to 2020.

There are 15 independent variables and apart from Sex ('male' and 'female') all others are continuous variables in float format (identifier in brackets):

- Sex
- Hemoglobin concentration in grams per decilitre - g/dL (hb)
- Pack cell volume/hematocrit in % (pcv)
- Red blood cell volume in 10^12/L (rbc)
- Mean cell volume in femtolitres-fl (mcv)
- Mean corpuscular hemoglobin in picograms-pg (mch)
- Mean corpuscular hemoglobin concentration in grams per decilitre-g/dL (mchc)
- Red blood cell distribution width in % (rdw)
- Total white blood cell count in 10^6/L (wbc) with white blood cell types in % - (neut, lymph)
- Total platelet count in 10^6/L (plt)
- Hemobglobin A, A2 and F in % from HPLC testing (hba, hba2, hbf)

The target variable is Phenotype, this is different in the two csv files:

- in alphanorm.csv: alpha carrier and normal (binary categorical)
- in twoalphas.csv: alpha trait and silent carrier (binary categorical)

I have chosen this dataset because as a health professional working in a laboratory it is very interesting to me. Predicting a phenotype based on blood parameters, in a disease that in fact causes alterations in blood parameters, could be a very helpful tool for laboratory clinitians. With the idea of changing the data as less as possible for ML, these datasets have three advantajes: it has mostly continuous variables(only require to convert the sex feature in numerical 0/1), there are only a few (2-3) empty cells in each dataset (which I fill with the mean), as I am using decision tree algorithms, I do not need scaling the data. So after minor changes we are ready to use both datasets.


### Baseline Model

There are [others](https://www.kaggle.com/code/letslive/alpha-thalassemia-classifier-1/notebook) models used previously in this precise dataset, but I choose Scikit-Learn’s Dummy Classifier model as the baseline model. One of the most important aspects of the model is recall, because we can handle a normal person classified has carrier, but we want avoid as much as posible an alpha carrier classified as normal. As the majority of samples are normal (alphanorm.csv) or silent carrier (twoalphas.csv), the dummy model in the prior strategy will cause 0 recall, easy to beat, so I set it in stratified mode.

### Algorithms

After dividing the dataset in train and test set with train_test_split, I started selecting the algorithm with the best performance in terms of recall and f1. For this I used StratifiedShuffleSplit to select the best Clasiffier for my dataset, and the algorithms tested were ExtraTreesClassifier, XGBClassifier, RandomForestClassifier, AdaBoostClassifier, and the Dummy Classifier. 

In this first approximation, the two datasets gave different results. In the first case (alpha carrier and normal phenotypes) Adaboost achieve the best scores, so I decide to use it in the next steps. 

![imagen](https://user-images.githubusercontent.com/115868725/201141137-be3171e2-206f-41b9-9a2d-cf3bc791dd0a.png)

I created validation curves to select the best hyperparameters for Adaboost, so I used validation curves with StratifiedKFold to select the best number of estimators and learning rate. 

![imagen](https://user-images.githubusercontent.com/115868725/201146621-294ed4cc-5fc3-42eb-8ca0-3e610f72163b.png)

![imagen](https://user-images.githubusercontent.com/115868725/201146681-7fef5231-3e6f-4e73-a060-f02ac61e090c.png)

Then I fit the Adaboost model with the best hyperparameters and I made a learning curve also with StratifiedKFold to check for possible overfittings. I plotted the confusion matrix for models (AdaBoost and Dummy) and finally, I calculated recall f1 and fbeta. 

![imagen](https://user-images.githubusercontent.com/115868725/201147783-edde74dc-114b-4b0d-9899-be9c8f4dfa25.png)

![imagen](https://user-images.githubusercontent.com/115868725/201147836-89fb25b9-34a1-4f8e-bed3-417314b99604.png)![imagen](https://user-images.githubusercontent.com/115868725/201147888-630fef34-b246-4e25-8fe2-ca6bd163f411.png)


On the other hand, with the second dataset (alpha trait and silent carrier phenotypes) ExtraTreesClassifier gets the highest scores, so I used this algorithm in the second dataset. 

![imagen](https://user-images.githubusercontent.com/115868725/201142382-5c56cfd0-3100-46a7-b27d-d1c450cc9122.png)

As with the first data set, I made validation curves to select the best max_depth and min_samples_split, and with the best options, I fit the model. 

![imagen](https://user-images.githubusercontent.com/115868725/201147002-7c648e1b-c6d4-43c1-b16d-842cdfd246f6.png)

![imagen](https://user-images.githubusercontent.com/115868725/201147072-73ba3d52-786e-46ba-9bbc-cbd7e4a9326e.png)

Then, I plotted a learning curve and confusion matrix as well, to finally evaluate the model with the chosen metrics.

![imagen](https://user-images.githubusercontent.com/115868725/201147471-2fa2ff39-2cb4-4547-87a3-1f3b57c21205.png)

![imagen](https://user-images.githubusercontent.com/115868725/201147566-da682e85-8ef6-43d2-aba1-445d9dabd651.png)![imagen](https://user-images.githubusercontent.com/115868725/201147628-5a63c0c7-0437-486f-b915-317b1ac83bfe.png)

At this point as the model performed well I decided to do a feature selection calculating the importance of each feature and plotting them in a graph. The lowest contributor was sex, so I created a new dataframe dropping the sex feature. After that, I decided to go further with a PCA. I need to scale the data with MinMaxScaler and split it into train and test first. Plotting the explained variance of the components, I saw that two components get most of the explained variation, so 

![imagen](https://user-images.githubusercontent.com/115868725/196921911-e4a57a39-ce53-475b-ad61-0f123efca922.png)


### Metrics

Both the Dummy classifier and the selected classifier models will be evaluated with the metrics recall, f1, and fbeta. In this kind of test, detecting all the not normal phenotypes is the main goal. It means as the fewer false negatives the better, even some false positives could happen. This is why have chosen recall as the first evaluation metric, f1 to see the global performance between recall and precision, and finally fbeta to see a whole vision but with recall predominance(fbeta=2). 

## Results and Analysis

In the first stage of screening to select the best algorithm, I found only small differences in both datasets. In the carrier/normal dataset, most models perform around f1 = 0.5-0.53, the same as the Dummy model. Adaboost rise 0.57 so that was the reason I selected it. In the alpha trait/silent carrier dataset, things were different and the models rise 0.65 as a minimum to 0.69 of ExtraTreeClassifier, compared with 0.5 of the dummy model. After selecting the best hyperparameters and fitting both models, in the first dataset the comparison with the dummy model in the recall, f1 and fbeta were: 

![imagen](https://user-images.githubusercontent.com/115868725/196921997-a4913975-d844-4922-beeb-4defd686691b.png)


In the second dataset the three metrics show the following results:

![imagen](https://user-images.githubusercontent.com/115868725/196922101-f7412b34-89ff-4726-bb73-d7958df8b983.png)

In the alpha trait/silent carrier dataset, after scaling the data and dropping the sex feature, the value of fbeta rises to 0.82. 

## Conclusion

The developed models have success beating the benchmark model, Scikit-Learn’s Dummy Classifier. Despite that, in the dataset of carrier/normal, the improvement is too slow, I consider it is far from being a usable model in real situations. Probably the differences between both groups are too small to build a good Classifier, and more data (adding more relevant features) is needed to improve the model. 
The second model is significantly better than the dummy model, with values in its metrics that make it a point of start for a model to implement in clinical practice. 

After PCA, the 2 principal components explain 83% of the variance. Surprises me that I made a scatterplot expecting to see two groups of dots of the two phenotypes.

![imagen](https://user-images.githubusercontent.com/115868725/196922299-ad4abbcf-dbb9-4d72-a5dc-d81b24cad1cb.png)![imagen](https://user-images.githubusercontent.com/115868725/196922414-3767f816-82c4-4ea5-aa26-16ebe54366e6.png)

This is probably the evidence that high variance at some features doesn’t imply a high significance for classification.







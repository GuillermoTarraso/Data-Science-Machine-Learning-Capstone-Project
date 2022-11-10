# Data Science & Machine Learning Capstone Project

This is my (Guillermo Tarrasó) capstone project for the Data Science & Machine Learning course from Coding Nomads.

The aim of this project is to develop and train two Machine Learning-based diagnostic classification models to differentiate Thalassemia Carrier states based on full blood count and/or Haemoglobin variants.

## How to use this Repository:
- Read the [Capstone Proposal](capstone_proposal_Thalassemia.md). This is a text file with the background and a brief description of the design of the project. 
- The file [Capstone Report](capstone_report.md) contains a summary of the methodology used in the project, the results and the conclusions.
- [Plotting](plotting.py) is an accessory file for some of the plots.

### Datasets
- You can find the first dataset with the normal and alpha carriers [here](alphanorm.xlsx)
- You can find the second dataset with the silent carriers and alpha triat [here](twoalphas.xlsx)

### Models:
- [Alphanorm Classifier](Thalassemia_alphanorm.ipynb) contains the code of the first model. Using the first dataset above we will differentiate between normal individuals and alpha thalassemia carriers. 
- [Twoalphas Classifier](Thalassemia_twoalphas.ipynb) contains the code of the second model. The second dataset will be used to differentiate between the two types of alpha carriers: silent carrier and alpha thalassemia trait, with the latter having more obvious clinical and biochemical changes and a higher risk of transmitting the disease to the next generation. 

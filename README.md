# abtest-mlops

**Table of content**

- [abtest-mlops](#abtest-mlops)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Features](#features)
    - [Data Exploration](#data-exploration)
    - [Classical A/B Testing](#classical-ab-testing)
    - [Sequential A/B Testing](#sequential-ab-testing)
    - [ML A/B Testing](#ml-ab-testing)
    - [Scripts](#scripts)
    - [Test](#test)
    - [Travis CI](#travis-ci)

## Overview
A/B testing allows comparing two or more versions of a given service against each other to find out which variation performs better. 

This repository contains an implementation of AB testing for the Classical, Sequential, and ML approaches. I have used data collected by an Advertising company running an online ad for a client to increase brand awareness. To increase its market competitiveness, the advertising company provides a further service that quantifies the increase in brand awareness as a result of the ads it shows to online users. 

The main objective is to design a reliable hypothesis testing algorithm to test if the ads that the advertising company runs resulted in a significant lift in brand awareness. Through this, we will explore Classical, Sequential, and ML approaches to A/B testing,

## Requirements
Python 3.5 and above, Pip and MYSQL
## Install
```
git clone https://github.com/eandualem/abtest-mlops
cd abtest-mlops
pip install -r requirements.txt
```
## Features

### Data Exploration
  - The notebook for Data Exploration is inside the notebooks folder in the file classical-ab-testing.ipynb.

### Classical A/B Testing
  - The notebook for Classical A/B Testing is inside the notebooks folder in the file classical-ab-testing.ipynb.

### Sequential A/B Testing
  - The notebook for Sequential A/B Testing is inside the notebooks folder in the file sequential-ab-testing.ipynb.

### ML A/B Testing
  - The notebook for sequential-ab-testing is inside the notebooks folder in the file ml-ab-testing.ipynb.

### Scripts
  - `create_dataset_versions`: simple script for creating different versions of the data AdSmartABdata.csv
  - `create_dataset`: simple script for creating train, test split of AdSmartABdata.csv
  - `create_features`: simple script for creating features for train and test data
  - `train_model`: class trains a model using 5-fold cross validation and returns the best model
  - `train_logistic_model`: simple script for training logistic regression using TrainModel class
  - `train_decision_trees_model`: simple script for training decision tree using TrainModel class
  - `train_xgboost_model`: simple script for training xgboost using TrainModel class
  - `evaluate_model`: class for calculates evaluation metrics for a give model using actual data
  - `evaluate_logistic_model`: simple script for evaluating logistic model using EvaluateModel class
  - `evaluate_decision_trees`: simple script for evaluating decision tree model using EvaluateModel class
  - `evaluate_xgboost_model`: simple script for evaluating xgboost model using EvaluateModel class
  - `df_helper`: helper class for reading csv and saving csv files

### Test
  - There are two tests inside the tests folder.

### Travis CI
  - The file .travis.yml contains the configuration for Travis.

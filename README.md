# income-analysis
A Data-Driven Approach to Predict whether an individual income would exceeds $50K based on education level, age, gender, occupation and race.

<a href="https://github.com/PyiThan/income-analysis" style="color: blue:">Click to see the full project report</a>

## Executive summary

This document outlines our approach to help government organizations overcome its challenge in effectively allocating its low income housing. Through the use of advanced machine learning techniques, we are confident in our ability to predict which individuals have an earning potential of less than 50 K. Our project involves identifying the most effective ML model for predicting responses and ensuring that our low income housing programs are personalized and targeted toward a person in need. To ensure the success of the project and provide reliable services to those in need, we aim for an accuracy score of 90 % or higher. This high threshold is essential to guarantee the effectiveness and reliability of our offerings.

## Introduction

The dataset, acquired from the University of California, Irvine Repository, from this project contains information about individuals income. It removed sensitive information. It was conducted by the U.S. Census. The ML techniques aim to enhance the government organizations to understand the predicting factors that potentially a classified “<=50K” or “>50K” response to a person's income. 

## Data Description

The data utilized in this project is sourced from the UCI Repository, specifically from the Adult Data Set. This dataset comprises 48,842 observations including 14 columns.

## Final Solution

I highly recommend utilizing the Random Forest algorithm as the chosen machine learning model, as it has proven to significantly enhance accuracy in various scenarios. However, considering the company’s focus on cost efficiency, I suggest implementing the K-nearest neighbor(KNN) algorithm. This method offers a favorable balance between accuracy and cost -effectiveness. With an accuracy rate of 71% Knn outperforms the logistic regression model and surpasses the results achieved by random guessing. By adopting KNN, the company can achieve reliable results while optimizing resource allocation.

## Programming Language use :book::

- Python
<p align="left">

<!-- For more icons please follow  https://github.com/MikeCodesDotNET/ColoredBadges -->
  <a href="#">
    <img 
src="https://raw.githubusercontent.com/MikeCodesDotNET/ColoredBadges/master/svg/dev/languages/python.svg" alt="python" style="vertical-align:top; margin:4px">
  </a>
  
  <a href="#">
    <img 
</p>

## Packages/Libraries
- pandas (1.5.3)
- matplotlib
- seaborn (0.12.2)
- scikit-learn 

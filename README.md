---
title: "Exercise effectiveness Prediction"
author: "Brad Martin"
date: "5/3/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(warning=FALSE,message=FALSE)
```

``` {r echo = FALSE}
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(parallel)
library(doParallel)
library(randomForest)

# enable parallel processing to speed up model evaluation
cluster<-makeCluster(detectCores()-1)
registerDoParallel(cluster)
set.seed(321)
```
## Introduction

A 2013 study by Velloso et al, aimed to assess the effectiveness or correctness of weight lifting exercises using data collected from body sensors. 3 axis acceleration, gyroscope and magnetometer data were recorded using 4 Inertial measurement units position on the test subject's glove, arm, belt and dumbbell as shown in figure 1. 6 participants performed sets of 10 dumbbell bicep curls in 5 classes of movement. Class A was correctly executed and Classes B - E incorrectly executed. 

The goal of this project is to predict how well 20 weight lifting exercises were performed from a test/validation data set. A model will be trained and evaluated using data from the original study data.

Figure 1. Sensor Positions
![Sensor Positions](http://groupware.les.inf.puc-rio.br/static/WLE/on-body-sensing-schema.png)

Reference:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Data Preparation  
The data used to train, test and validate each model was provided in the Coursera project instructions:  

Training data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
Validation data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Data from the original study can be found at http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.

```{r echo=FALSE}
df.training <-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
df.testing <-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

Exploratory analysis was performed on the training and validation data sets. 

The training data contains `r nrow(df.training)` rows and `r ncol(df.training)` columns records. 

Examination of the training and test data sets using str() showed many columns missing data or contain NA value. These columns were removed as well as rows containing NA. Meta data identifying each participant was removed was it is irrelevant for modeling.
As the data set was too large for any further practical analysis and without expertise in the specifics of the measurements, the data set could not be reduced further and all remaining columns were used for modeling. The *classe* variable was converted to a factor variable. 

``` {r}
# convert classe variable to a factor
df.training$classe <-as.factor(df.training$classe)

# Remove columns and rows containing null and NA valves from Training data
df.training<-df.training %>%select_if(~!any(is.na(.) || is.null(.) || . =="")) %>% drop_na()
# remove meta data not required for modelling
df.training <- select(df.training,-c(1:7))

# Remove columns and rows containing null and NA valves from Training data
df.testing<- df.testing %>%select_if(~!any(is.na(.)|| is.null(.) || . =="")) %>% drop_na()
# remove meta data not required for modelling
df.testing <- select(df.testing,-c(1:7))
```

The training data set then was split into 70% training and 30% test data:
```{r}
inTrain <- createDataPartition(df.training$classe,p=0.7,list=FALSE)
sub.training <- df.training[inTrain,]
sub.testing <- df.training[-inTrain,]
```

## Modelling

In order to determine the best model Random Forest, Regression Tree and Boosting models were trained and evaluated. A Random Forest model with Principal Component Analysis was also trained to determine if a deduced set of features would improve results. Each model was trained using a 5 fold cross validation sampling.

The following is an example of the modeling and validation code:

```{r cache=TRUE}
#Build a Random Forest model with 5 fold Cross Validation

  #Set training control parameters 
  tc1<- trainControl(method="cv",number=5,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod1_t <- system.time(modfit1<-train(classe ~ .,data=sub.training, method="rf",trControl=tc1))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr1<-predict(modfit1,sub.training)
  cm_tr1<-confusionMatrix(sub.training$classe,pred_tr1)
  
  #Evaluate accuracy of model predictions on test data
  pred1<-predict(modfit1,sub.testing)
  cm1<-confusionMatrix(sub.testing$classe,pred1)

```

When the models were evaluated against the unseen Test data set, the Random Forest with 5 fold Cross Validation had the lowest out of sample error (most accurate).

A further evaluation of the Random Forest model with 10 and 15 fold Cross validation show reduced accuracy.

``` {R cache=TRUE, echo = FALSE, results='hide'}

#Build a Random Forest model with 5 fold Cross Validation and Princpal Component Analysis
  #Set training control parameters
  tc2<- trainControl(method="cv",number=5,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod2_t <- system.time(modfit2<-train(classe ~ .,data=sub.training, method="rf",trControl=tc2,preProcess="pca"))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr2<-predict(modfit2,sub.training)
  cm_tr2<-confusionMatrix(sub.training$classe,pred_tr2)
  
  #Evaluate accuracy of model predictions on test data
  pred2<-predict(modfit2,sub.testing)
  cm2<-confusionMatrix(sub.testing$classe,pred2)

#Build a Regression Tree model with 5 fold Cross Validation
  #Set training control parameters
  tc3 <- trainControl(method="cv",number=5,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod3_t<-system.time(modfit3 <-train(classe ~ .,data=sub.training,method="rpart",trControl=tc3))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr3<-predict(modfit3,sub.training)
  cm_tr3<-confusionMatrix(sub.training$classe,pred_tr3)
  
  #Evaluate accuracy of model predictions on test data
  pred3 <- predict(modfit3,sub.testing)
  cm3<-confusionMatrix(sub.testing$classe,pred3)

#Build a Boosting model with 5 fold Cross Validation
  #Set training control parameters
  tc4 <- trainControl(method="cv",number=5,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod4_t<-system.time(modfit4 <-train(classe ~ .,data=sub.training,method="gbm",trControl=tc4))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr4<-predict(modfit4,sub.training)
  cm_tr4<-confusionMatrix(sub.training$classe,pred_tr4)
  
  #Evaluate accuracy of model predictions on test data
  pred4 <- predict(modfit4,sub.testing)
  cm4<-confusionMatrix(sub.testing$classe,pred4)

#Build a Random Forest model with 10 fold Cross Validation
  #Set training control parameters
  tc5 <- trainControl(method="cv",number=10,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod5_t <- system.time(modfit5<-train(classe ~ .,data=sub.training, method="rf",trControl=tc5))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr5<-predict(modfit5,sub.training)
  cm_tr5<-confusionMatrix(sub.training$classe,pred_tr5)
  
  #Evaluate accuracy of model predictions on test data
  pred5 <-predict(modfit5,sub.testing)
  cm5<-confusionMatrix(sub.testing$classe,pred5)

#Build a Random Forest model with 15 fold Cross Validation
  #Set training control parameters
  tc6<- trainControl(method="cv",number=15,allowParallel = TRUE)
  
  #Create the model and evaluate the time to complete
  mod6_t <- system.time(modfit6<-train(classe ~ .,data=sub.training, method="rf",trControl=tc6))
  
  #Evaluate accuracy of model predictions on training data
  pred_tr6<-predict(modfit6,sub.training)
  cm_tr6<-confusionMatrix(sub.training$classe,pred_tr6)
  
  #Evaluate accuracy of model predictions on test data
  pred6<-predict(modfit6,sub.testing)
  cm6<-confusionMatrix(sub.testing$classe,pred6)

```

Table 1. Model Results
``` {r echo=FALSE}

Model <- c("Random Forest - 5 Fold Cross Validation",
            "Random Forest - 5 Fold Cross Vaidation with PCA",
                   "Regression Tree - 5 Fold Cross Vaidation",
                   "Boosted - 5 Fold Cross Vaidation",
                   "Random Forest - 10 Fold Cross Vaidation",
                   "Random Forest - 15 Fold Cross Vaidation")
TrainingAccuracy<- c(cm_tr1$overall[[1]],
                      cm_tr2$overall[[1]],
                      cm_tr3$overall[[1]],
                      cm_tr4$overall[[1]],
                      cm_tr5$overall[[1]],
                      cm_tr6$overall[[1]])


TestAccuracy<- c(cm1$overall[[1]],
                    cm2$overall[[1]],
                    cm3$overall[[1]],
                    cm4$overall[[1]],
                    cm5$overall[[1]],
                    cm6$overall[[1]])

ProcessTime <- c(mod1_t[[3]],
                         mod2_t[[3]],
                          mod3_t[[3]],
                          mod4_t[[3]],
                          mod5_t[[3]],
                          mod6_t[[3]])

Results <- data.frame(Model,TrainingAccuracy,TestAccuracy,ProcessTime)
knitr::kable(Results)

```


The confusion matrix for Random Forest with 5 Fold Cv:

``` {r echo=FALSE}
cm1$table

```


```{r echo=FALSE}
#stop parallel processing
stopCluster(cluster)
registerDoSEQ()
```

## Conclusion
A Random Forest model with 5 fold CV was trained to predict the assess the correctness of the weight lifting exercises with `r cm1$overall[[1]]` accuracy. This model will be used to evaluate the Test/Evaluation data set.

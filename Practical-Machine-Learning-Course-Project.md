Exercise effectiveness Prediction
================
Brad Martin
5/3/2021

## Introduction

A 2013 study by Velloso et al, aimed to assess the effectiveness or
correctness of weight lifting exercises using data collected from body
sensors. 3 axis acceleration, gyroscope and magnetometer data were
recorded using 4 Inertial measurement units position on the test
subject’s glove, arm, belt and dumbbell as shown in figure 1. 6
participants performed sets of 10 dumbbell bicep curls in 5 classes of
movement. Class A was correctly executed and Classes B - E incorrectly
executed.

The goal of this project is to predict how well 20 weight lifting
exercises were performed from a test/validation data set. A model will
be trained and evaluated using data from the original study data.

Figure 1. Sensor Positions ![Sensor
Positions](http://groupware.les.inf.puc-rio.br/static/WLE/on-body-sensing-schema.png)

Reference: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks,
H. Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Data Preparation

The data used to train, test and validate each model was provided in the
Coursera project instructions:

Training data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>
Validation data:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

Data from the original study can be found at
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.

Exploratory analysis was performed on the training and validation data
sets.

The training data contains 19622 rows and 160 columns records.

Examination of the training and test data sets using str() showed many
columns missing data or contain NA value. These columns were removed as
well as rows containing NA. Meta data identifying each participant was
removed was it is irrelevant for modeling. As the data set was too large
for any further practical analysis and without expertise in the
specifics of the measurements, the data set could not be reduced further
and all remaining columns were used for modeling. The *classe* variable
was converted to a factor variable.

``` r
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

The training data set then was split into 70% training and 30% test
data:

``` r
inTrain <- createDataPartition(df.training$classe,p=0.7,list=FALSE)
sub.training <- df.training[inTrain,]
sub.testing <- df.training[-inTrain,]
```

## Modelling

In order to determine the best model Random Forest, Regression Tree and
Boosting models were trained and evaluated. A Random Forest model with
Principal Component Analysis was also trained to determine if a deduced
set of features would improve results. Each model was trained using a 5
fold cross validation sampling.

The following is an example of the modeling and validation code:

``` r
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

When the models were evaluated against the unseen Test data set, the
Random Forest with 5 fold Cross Validation had the lowest out of sample
error (most accurate).

A further evaluation of the Random Forest model with 10 and 15 fold
Cross validation show reduced accuracy.

Table 1. Model Results

| Model                                           | TrainingAccuracy | TestAccuracy | ProcessTime |
|:------------------------------------------------|-----------------:|-------------:|------------:|
| Random Forest - 5 Fold Cross Validation         |        1.0000000 |    0.9937128 |      213.38 |
| Random Forest - 5 Fold Cross Vaidation with PCA |        1.0000000 |    0.9774002 |       97.14 |
| Regression Tree - 5 Fold Cross Vaidation        |        0.4990173 |    0.4890399 |        3.33 |
| Boosted - 5 Fold Cross Vaidation                |        0.9740846 |    0.9610875 |       60.69 |
| Random Forest - 10 Fold Cross Vaidation         |        1.0000000 |    0.9933730 |      451.20 |
| Random Forest - 15 Fold Cross Vaidation         |        1.0000000 |    0.9933730 |      681.16 |

The Confusion Matrix for Random Forest with 5 Fold Cv:

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    2    0    0    0
    ##          B    4 1132    3    0    0
    ##          C    0    9 1014    3    0
    ##          D    0    0   14  950    0
    ##          E    0    0    0    2 1080

## Conclusion

A Random Forest model with 5 fold CV was trained to predict the assess
the correctness of the weight lifting exercises with 0.9937128 accuracy.
This model will be used to evaluate the Test/Evaluation data set.

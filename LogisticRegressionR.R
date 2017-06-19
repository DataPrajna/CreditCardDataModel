#LogisticRegression for Kaggle data

library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library(gtools) # for discretisation
library(corrplot)
library(Hmisc)
library(devtools)
library(PerformanceAnalytics)
library(FactoMineR)

#Load data from csv file
data <- read.csv("C:/DataAnalysisOperantAI/UCI_Credit_Card.csv")

#check dimesion of data
dim(data)

#check for missing values
library(Amelia) 
sapply(data, function(x) sum(is.na(x)))
missmap(data, main="Missing values Vs Observed values") #for visualization

#Model Fitting
train_data_size = 0.75*(nrow(data))
train_data = data[1:train_data_size,]
test_data_index_start = train_data_size+1
test_data_index_end = nrow(data)
test_data  =  data[test_data_index_start:test_data_index_end,]

glmModel <-glm(default.payment.next.month ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE +
                 AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 +
                 BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 +
                 PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6,
                       family=binomial(link="logit"),data=train_data)

summary(glmModel)

#Training accuracy

fitted_results_training <- glmModel$fitted.values
fitted_results_training <- ifelse(fitted_results_training > 0.5, 1,0)
training_data_missclassified <- mean(fitted_results_training  != train_data$default.payment.next.month)
print(paste('Training Accuracy', 1 - training_data_missclassified))


#Prediction 

fitted_results_testing <- predict(glmModel, newdata = test_data, type = 'response')
fitted_results_testing <- ifelse(fitted_results_testing > 0.5,1,0)
testing_data_missclassified <- mean(fitted_results_testing != test_data$default.payment.next.month)
print(paste(' Testing Accuracy', 1-testing_data_missclassified))

#ROC Curve Testing, auc > 0.5, good prediction rate

library(ROCR)
p <- predict(glmModel, newdata=test_data, type="response")
pr <- prediction(p, test_data$default.payment.next.month)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, lwd = 2, main = "Area Under Curve")

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
print(paste('Area under Curve', auc))









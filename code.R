setwd("C:\\R Programming\\WNS")
train<-read.csv("train.csv",na.strings = c(""," "))
test<-read.csv("test.csv",na.strings = c(""," "))
train$employee_id<-NULL
region<-train$region
#train$region<-NULL

library(ggplot2)
library(dplyr)
library(ROSE)

explore<-train %>% group_by(department) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(education) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(gender) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(recruitment_channel) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(no_of_trainings) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(previous_year_rating) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(length_of_service) %>% summarise(mean(is_promoted)) ## make groups and see

explore<-train %>% group_by(KPIs_met..80.) %>% summarise(mean(is_promoted))

explore<-train %>% group_by(awards_won.) %>% summarise(mean(is_promoted))



##EDA
g<-ggplot(train,aes(x=age,fill=as.factor(is_promoted)))+geom_histogram(stat="bin",bins = 10)         
print(g)

g<-ggplot(train,aes(x=length_of_service,fill=as.factor(is_promoted)))+geom_histogram(stat="bin",bins = 30)         
print(g)


##Feature transformation
train$servicegroups<-cut(train$length_of_service,breaks = c(1,6,11,21,31,40),include.lowest = T)
levels(train$servicegroups)<-c("New","Experienced","Old","VeryOld","Retired")
explore<-train %>% group_by(servicegroups) %>% summarise(mean(is_promoted))

train$agegroups<-cut(train$age,breaks = c(19,33,48,61),include.lowest = T)
levels(train$agegroups)<-c("Young","Middle","Old")
explore<-train %>% group_by(agegroups) %>% summarise(mean(is_promoted))


##Feature engineering
train$istraininggrt6<-factor(ifelse(train$no_of_trainings>6,0,1))

##Imputing mean value to NAs in previous_year_rating
train$previous_year_rating[is.na(train$previous_year_rating)]<-mean(train$previous_year_rating,na.rm = T)

train$is_promoted<-as.factor(train$is_promoted)


##Renaming departments and Education
train$department<-gsub("&","n",train$department)
train$department<-as.factor(gsub(" ","",train$department))

train$education<-as.character(train$education)
train$education[is.na(train$education)]<-"Missing"
train$education<-gsub("&","n",train$education)
train$education<-gsub("'","",train$education)
train$education<-gsub(" ","",train$education)
train$education<-as.factor(gsub(" ","",train$education))


library(dummies)
library(caret)
newtrain<-dummy.data.frame(train[,-13])

#normalise<-function(x){
#  (x-min(x))/(max(x)-min(x))
#}
#newtrain<-sapply(newtrain,normalise)
kmeans<-kmeans(newtrain,centers = 2)
cluster<-kmeans$cluster
cluster1<-subset(train,cluster==1)
cluster2<-subset(train,cluster==2)
newtrain$cluster<-cluster

library(flexclust)
kcca<-kcca(newtrain,k=2)

newtrain$is_promoted<-train$is_promoted


under<-ovun.sample(is_promoted~.,data=newtrain,method="under",p=0.09)$data


trControl<-trainControl(method="cv",number = 5)
modelrpart<-train(is_promoted~.,data=under,method="rpart",trControl=trControl)
predictionsrpart<-predict(modelrpart,under,type="prob")
predictionsrpart<-predictionsrpart[,2]
table(predictionsrpart>0.5,under$is_promoted)

model<-train(is_promoted~.,data=under,method="glm",trControl=trControl)
predictionsglm<-predict(modelglm,under,type="prob")
predictionsglm<-predictionsglm[,2]
table(predictionsglm>0.5,under$is_promoted)


modelbayes<-train(is_promoted~.,data=under,method="bayesglm",trControl=trControl)
predictionsbayes<-predict(modelbayes,under,type="prob")
predictionsbayes<-predictionsbayes[,2]
table(predictionsbayes>0.5,under$is_promoted)

modelctree<-train(is_promoted~.,data=under,method="ctree",trControl=trControl)
predictionsctree<-predict(modelctree,under)
table(predictionsctree,under$is_promoted)


library("xgboost")
library("Matrix")
data_variables <- as.matrix(under[,-71])
data_label <- as.numeric(under[,"is_promoted"])-1
data_matrix <- xgb.DMatrix(data = data_variables, label = data_label)


xgb_params <- list(booster = "gbtree", objective = "binary:logistic",
                   eta=0.1, max_depth=10)
xgbcv <- xgb.cv( params = xgb_params, data = data_matrix, nrounds = 100, nfold = 10, showsd = T, 
                 stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)


nround    <- xgbcv$best_iteration # number of XGBoost rounds
cv.nfold  <- 10

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
bst_model <- xgb.train(params = xgb_params,
                       data = data_matrix,
                       nrounds = nround)
test_matrix<-xgb.DMatrix(data = as.matrix(under[,-71]))
predictionsXGBoost<-predict(bst_model,test_matrix)
table(predictionsXGBoost>0.5,under$is_promoted)



library(h2o)
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-10.0.2')
h2o.init()



h2otrain<-as.h2o(under)
h2otest<-as.h2o(under)
modeldeeplearning<-h2o.deeplearning(x = 1:70 ,
                                    y = "is_promoted",
                                    training_frame = h2otrain,
                                    activation = "RectifierWithDropout",
                                    l1 = 1.0e-5,l2 = 1.0e-5,
                                    hidden=c(80,80),
                                    epochs = 150,
                                    seed = 3.656455e+18)

h2opredictions<-as.data.frame(h2o.predict(modeldeeplearning,h2otest))
h2opredictions<-h2opredictions$p1
table(h2opredictions>0.5,under$is_promoted)


######################
######################
#Transforming test set

##Feature transformation
test$servicegroups<-cut(test$length_of_service,breaks = c(1,6,11,21,31,40),include.lowest = T)
levels(test$servicegroups)<-c("New","Experienced","Old","VeryOld","Retired")


test$agegroups<-cut(test$age,breaks = c(19,33,48,61),include.lowest = T)
levels(test$agegroups)<-c("Young","Middle","Old")



##Feature engineering
test$istraininggrt6<-factor(ifelse(test$no_of_trainings>6,0,1))

##Imputing mean value to NAs in previous_year_rating
test$previous_year_rating[is.na(test$previous_year_rating)]<-mean(test$previous_year_rating,na.rm = T)



##Renaming departments and Education
test$department<-gsub("&","n",test$department)
test$department<-as.factor(gsub(" ","",test$department))

test$education<-as.character(test$education)
test$education[is.na(test$education)]<-"Missing"
test$education<-gsub("&","n",test$education)
test$education<-gsub("'","",test$education)
test$education<-gsub(" ","",test$education)
test$education<-as.factor(gsub(" ","",test$education))


library(dummies)
library(caret)
newtest<-dummy.data.frame(test)

newtest$cluster<-predict(kcca,newtest)

#library(keras)
#newtest<-as.matrix(newtest)

h2otest<-as.h2o(newtest)

h2opredictions<-as.data.frame(h2o.predict(modeldeeplearning,h2otest))
h2opredictions<-h2opredictions$predict

output<-data.frame(employee_id=test$employee_id,is_promoted=h2opredictions)
write.csv(output,"output.csv",row.names = FALSE)



setwd("C:/Users/asus/Desktop/M2_stat/Projet_MachineLearning")

library(plyr)
library(dplyr)
library(glmnet)
library(caret)
library(MASS)
library(randomForest)
library(kernlab)
library(FactoMineR)
library(lubridate)
library(leaps)
library(bcv)
library(ggplot2)
library(e1071)
library(gbm)
library(doParallel)
library(bst)
library(Metrics)
library(mlbench)
library(xgboost)
library(ggmap)

######################
# IMPORT DES DONNEES #
######################

train0<-read.csv2("train_Hgeo.csv",header=TRUE,sep=";",dec=".")
test<-read.csv2("test_Hgeo.csv",header=TRUE,sep=";",dec=".")
testb<-read.csv2("test.csv",header=TRUE,sep=";")

sapply(train0, class)
sapply(test, class)
sapply(testb, class)
summary(test)
summary(testb)
###########################
# CONVERTIR LES VARIABLES #
###########################

train0$ddH10_rose4<-as.factor(as.character(train0$ddH10_rose4))
train0$date <- as.Date(train0$date,format = "%Y-%m-%d")
train0$insee <- as.factor(as.character(train0$insee))

test$ddH10_rose4<-as.factor(as.character(test$ddH10_rose4))
test$date <- as.Date(test$date,format = "%d/%m/%Y")
test$insee <- as.factor(as.character(test$insee))

######################################
# Suppression des variables: lon lat # 
######################################

train0<-dplyr::select(train0,-c(lon,lat))
test<-dplyr::select(test,-c(lon,lat))

######################################
# Creation des indicateurs par ville #
######################################

#train
train0$ind_Nice<-ifelse(train0$insee=="6088001",1,0)
train0$ind_Toulouse<-ifelse(train0$insee=="31069001",1,0)
train0$ind_Bordeaux<-ifelse(train0$insee=="33281001",1,0)
train0$ind_Rennes<-ifelse(train0$insee=="35281001",1,0)
train0$ind_Lille<-ifelse(train0$insee=="59343001",1,0)
train0$ind_Strasbourg<-ifelse(train0$insee=="67124001",1,0)
train0$ind_Paris<-ifelse(train0$insee=="75114001",1,0)

#test
test$ind_Nice<-ifelse(test$insee=="6088001",1,0)
test$ind_Toulouse<-ifelse(test$insee=="31069001",1,0)
test$ind_Bordeaux<-ifelse(test$insee=="33281001",1,0)
test$ind_Rennes<-ifelse(test$insee=="35281001",1,0)
test$ind_Lille<-ifelse(test$insee=="59343001",1,0)
test$ind_Strasbourg<-ifelse(test$insee=="67124001",1,0)
test$ind_Paris<-ifelse(test$insee=="75114001",1,0)

sapply(train0,class)
sapply(test,class)

train0_ok<-train0[,-c(1,2,20)]

index<-sample(1:nrow(train0_ok),0.6*nrow(train0_ok),replace=FALSE)
train<-na.omit(train0_ok[c(index),])
validation<-na.omit(train0_ok[-c(index),])

################
# gbm gaussien #
################

# le 07/12/2017 a 23h 30 
model_gbm<- gbm(tH2_obs~.,data=train,distribution="gaussian",n.trees=40000,interaction.depth=5,cv.folds=10,shrinkage=0.01)

plot(model_gbm$cv.error,type="l")
title("L'erreur estimée en fonction du nombre d'arbres sur l'échantillon d'apprentissage")

B<- gbm.perf(model_gbm,method="cv")

pred_gbm<- predict(model_gbm,newdata=validation,n.trees=B)
sqrt(mean((pred_gbm-validation$tH2_obs)**2)) #1.006

pred_gbm2 <- predict(model_gbm,newdata=validation,n.trees=30000)
sqrt(mean((pred_gbm2-validation$tH2_obs)**2)) #1.02

pred_gbm3 <- predict(model_gbm,newdata=validation,n.trees=25000)
sqrt(mean((pred_gbm3-validation$tH2_obs)**2)) #1.03

pred_gbm4 <- predict(model_gbm,newdata=validation,n.trees=20000)
sqrt(mean((pred_gbm4-validation$tH2_obs)**2)) #1.04

pred_gbm5 <- predict(model_gbm,newdata=validation,n.trees=15000)
sqrt(mean((pred_gbm5-validation$tH2_obs)**2)) #1.056

pred_gbm6 <- predict(model_gbm,newdata=validation,n.trees=10000)
sqrt(mean((pred_gbm6-validation$tH2_obs)**2)) #1.07

#Fausse erreur
pred_test<-predict(model_gbm,newdata=test,n.trees=B)
sqrt(mean((pred_test-test$tH2)**2)) #1.30

pred_test2<-predict(model_gbm,newdata=test,n.trees=30000)
sqrt(mean((pred_test2-test$tH2)**2)) #1.28

pred_test3<-predict(model_gbm,newdata=test,n.trees=25000)
sqrt(mean((pred_test3-test$tH2)**2)) #1.27

pred_test4<-predict(model_gbm,newdata=test,n.trees=20000)
sqrt(mean((pred_test4-test$tH2)**2)) #1.25

pred_test5<-predict(model_gbm,newdata=test,n.trees=15000)
sqrt(mean((pred_test5-test$tH2)**2)) #1.23


test$tH2_obs<-pred_test5
write.table(test,"SOUMISSION_40.csv",sep=";",dec=".",row.names=FALSE) #Avec 15000

test$tH2_obs<-pred_test3
write.table(test,"SOUMISSION_41.csv",sep=";",dec=".",row.names=FALSE) #Avec 25000

test$tH2_obs<-pred_test4
write.table(test,"SOUMISSION_42.csv",sep=";",dec=".",row.names=FALSE) #Avec 20000

liste_nb <- c()
liste_err <- c()
for (nb in seq(100,40000,100)){
  pred_gbmb <- predict(model_gbm,newdata=validation,n.trees=nb)
  err <- sqrt(mean((pred_gbmb-validation$tH2_obs)**2))
  liste_nb <- c(liste_nb,nb)
  liste_err <- c(liste_err,err)
}

plot(liste_err~liste_nb,type="l")
title("L'erreur estimée en fonction du nombre d'arbres sur l'échantillon de validation")

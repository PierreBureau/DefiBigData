# DefiBigData

# install.packages(c("lubridate", "magrittr"))
library("lubridate")
library("magrittr")
# install.packages("missForest")
library(missForest)
library(bcv)
library(xgboost)
library(Matrix)
library(dplyr)

setwd("/home/bureau_p/Master 2/defi/")

####
# OUVERTURE FICHIERS TRAIN
####

fichiers=list.files('/home/bureau_p/Master 2/defi/data_meteo/')
fichiers <- sort(fichiers)
n=length(fichiers)

for (i in 1:n) {
  file <- read.csv2(paste("/home/bureau_p/Master 2/defi/data_meteo/train_",i,".csv",sep=""),header=TRUE,sep=";",colClasses=c(insee="factor",date="Date",ech="factor"))
  assign(paste("train_",i, sep = ""),file)
}

####
# OUVERTURE FICHIER TEST
####

test<-read.csv2("test.csv",header=TRUE,sep=";",colClasses=c(insee="factor",date="Date",ech="factor"))

test$flvis1SOL0<-as.numeric(test$flvis1SOL0)

####
# RASSEMBLER FICHIERS TRAIN
####

train0 <- do.call("rbind", mget(ls(pattern='train_')))

train0$flvis1SOL0<-as.numeric(train0$flvis1SOL0)

######################
# IMPUTATION DES NA ##
######################

summary(train0)
sapply(train0,class)

# Varibales contenant des NA'S :
# th2_obs = 20
# capeinsSol0 = 18788
# ciwcH20 = 973
# clwlH20 = 4480
# ...

# Variables contenant pas de NA'S
# date, insee, ddh10_rose4,ech,mois

summary(test)

# Variables contenant des NA'S
# flir1Sol0 = 98
# fllat1SOl0 = 98
# flsen1SOl0 = 98
# rr1SOl0 = 2940

# CERATION DE L'ALGORITHME D'IMPUTATION SUR LE JEU TEST et TRAIN

# Il existe différentes méthodes, on choisit d'abord la SVD

train0$jour <- yday(train0$date)

# install.packages("bcv")

test<-read.csv2("test3.csv",header=TRUE,sep=";",dec=".")

test$date <- as.Date(test$date, format = "%d/%m/%Y")

test$ddH10_rose4<-as.factor(as.character(test$ddH10_rose4))
levels(test$ddH10_rose4)<-c("1.0","2.0","3.0","4.0")

test$flvis1SOL0<-as.numeric(as.character(test$flvis1SOL0))
test$ddH10_rose4<-as.factor(as.character(test$ddH10_rose4))


# IMPUTATION SUR LE FICHIER TEST (SVD)

test2<-test[,c(8,9,10,11)]
dtest<-impute.svd(test2,k=3,maxiter=1000)$x
colnames(dtest)<-colnames(test2)
testnew<-cbind(test[,-c(8,9,10,11)],dtest)


# IMPUTATION SUR LE FICHIER TRAIN (SVD)
train2<-train0[,-c(1,2,7,31,30)]
dtrain<-impute.svd(train2,k=2,maxiter=1000)$x
colnames(dtrain)<-colnames(train2)
trainnew<-cbind(train0[,c(1,2,7,31,30)],dtrain)
summary(trainnew)

# IMPUTATION SUR LE FICHIER TRAIN (missForest)
train2<-train0[,-c(1,2,7,31,30)]
dtrain<-missForest(train2,maxiter=10,
                           ntree = 200, variablewise = TRUE)$ximp


# MAJ des variables de TEST

testnew$jour<-yday(testnew$date)
testnew$insee<-as.factor(as.character(testnew$insee))
testnew$ech<-as.factor(as.character(testnew$ech))

trainnew$insee<-as.factor(as.character(trainnew$insee))
trainnew$ech<-as.factor(as.character(trainnew$ech))



# REGRESSION SUR AVEC LE NEWTRAIN AYANT EFFECTUE UNE IMPUTATION

index<-sample(1:nrow(trainnew),0.60*nrow(trainnew),replace=FALSE)
train<-trainnew[c(index),]
validation<-trainnew[-c(index),]

reg<-lm(tH2_obs~.,data=train)
prediction<-predict(reg,newdata=validation)
sqrt(mean((prediction-validation[,6])**2))

# REGRESSION SUR AVEC LE NEWTRAIN SANS EFFECTUE UNE IMPUTATION

index<-sample(1:nrow(train0),0.60*nrow(train0),replace=FALSE)
train<-na.omit(train0[c(index),])
validation<-na.omit(train0[-c(index),])

reg<-lm(tH2_obs~.,data=train)
prediction<-predict(reg,newdata=validation)
sqrt(mean((prediction-validation[,3])**2))


############
# SEPARATION DES VILLES 
############

doublonstest<-which(duplicated(train0$insee))
villes<-as.character(train0$insee[-doublonstest])


for (city in villes){
    fichier <- subset(train0,insee == city)
    assign(paste(city,"_",sep = ""),fichier[,-c(20,2,1)])
    
}

summary(`6088001_`)
summary(train0)

# REGRESSION SUR LA PREMIERE VILLE 6088001

############
villes[1] ##
############

index<-sample(1:nrow(`6088001_`),0.60*nrow(`6088001_`),replace=FALSE)
train<-na.omit(`6088001_`[c(index),])
validation<-na.omit(`6088001_`[-c(index),])

reg1<-lm(tH2_obs~.,data=train)
prediction<-predict(reg1,newdata=validation)
err1 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,3,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,3,28)]


library(caret)
vignette("caretTrain", package="caret")

# Fitting model
sqrt(NCOL(train))
summary(train)
sapply(train,class)


gbmGrid <-  expand.grid(interaction.depth = c(2,4,6,9),
                        n.trees = c(10,20,40), 
                        shrinkage = c(0.05,0.1,0.2),
                        n.minobsinnode = 10)

# interaction.depth = 1 : additive model, interaction.depth = 2 : two-way interactions, etc. 

fitControl <- trainControl( method = "repeatedcv", number = 5, repeats = 4)

fit1 <- train(tH2_obs ~ ., data = train, 
              distribution = "gaussian",
              method = "gbm", 
              holdout.fraction=2/5,
              #trControl = fitControl,
              tuneGrid = gbmGrid,
              importance = TRUE)

predicted= predict(fit1,validation) 
err1 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)

end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
             )

ntrees = 5000

# library(gbm)
model1 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model1,newdata = validation[,-1],
                   n.trees = gbm.perf(model1, plot.it = FALSE)
                   ) 

err1 <- sqrt(mean((predicted-validation[,1])**2))

gbm.perf(model)
# 2349


############
villes[2] ##
############

index<-sample(1:nrow(`31069001_`),0.60*nrow(`31069001_`),replace=FALSE)
train<-na.omit(`31069001_`[c(index),])
validation<-na.omit(`31069001_`[-c(index),])

reg2<-lm(tH2_obs~.,data=train)
prediction<-predict(reg2,newdata=validation)
err2 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit2 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit2,validation) 
err2 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model2 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model2,newdata = validation[,-1],
                   n.trees = gbm.perf(model2, plot.it = FALSE)
) 

err2 <- sqrt(mean((predicted-validation[,1])**2))

############
villes[3] ##
############

index<-sample(1:nrow(`33281001_`),0.60*nrow(`33281001_`),replace=FALSE)
train<-na.omit(`33281001_`[c(index),])
validation<-na.omit(`33281001_`[-c(index),])

reg3<-lm(tH2_obs~.,data=train)
prediction<-predict(reg3,newdata=validation)
err3 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit3 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit3,validation) 
err3 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model3 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)

predicted= predict(object = model3,newdata = validation[,-1],
                   n.trees = gbm.perf(model3, plot.it = FALSE)
) 

err3 <- sqrt(mean((predicted-validation[,1])**2))

############
villes[4] ##
############

index<-sample(1:nrow(`35281001_`),0.60*nrow(`35281001_`),replace=FALSE)
train<-na.omit(`35281001_`[c(index),])
validation<-na.omit(`35281001_`[-c(index),])

reg4<-lm(tH2_obs~.,data=train)
prediction<-predict(reg4,newdata=validation)
err4 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit4 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit4,validation) 
err4 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model4 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model4,newdata = validation[,-1],
                   n.trees = gbm.perf(model4, plot.it = FALSE)
) 

err4 <- sqrt(mean((predicted-validation[,1])**2))

############
villes[5] ##
############

index<-sample(1:nrow(`59343001_`),0.60*nrow(`59343001_`),replace=FALSE)
train<-na.omit(`59343001_`[c(index),])
validation<-na.omit(`59343001_`[-c(index),])

reg5<-lm(tH2_obs~.,data=train)
prediction<-predict(reg5,newdata=validation)
err5 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit5 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit5,validation) 
err5 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model5 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model5,newdata = validation[,-1],
                   n.trees = gbm.perf(model5, plot.it = FALSE)
) 

err5 <- sqrt(mean((predicted-validation[,1])**2))

############
villes[6] ##
############

index<-sample(1:nrow(`67124001_`),0.60*nrow(`67124001_`),replace=FALSE)
train<-na.omit(`67124001_`[c(index),])
validation<-na.omit(`67124001_`[-c(index),])

reg6<-lm(tH2_obs~.,data=train)
prediction<-predict(reg6,newdata=validation)
err6 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boostingt
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit6 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit6,validation) 
err6 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model6 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model6,newdata = validation[,-1],
                   n.trees = gbm.perf(model6, plot.it = FALSE)
) 

err6 <- sqrt(mean((predicted-validation[,1])**2))

############
villes[7] ##
############

index<-sample(1:nrow(`75114001_`),0.60*nrow(`75114001_`),replace=FALSE)
train<-na.omit(`75114001_`[c(index),])
validation<-na.omit(`75114001_`[-c(index),])

reg7<-lm(tH2_obs~.,data=train)
prediction<-predict(reg7,newdata=validation)
err7 <- sqrt(mean((prediction-validation[,1])**2))

####
# Méthode boosting
####

train$ech<-as.numeric(as.character(train$ech))
train <- train[,-c(5,28)]

validation$ech<-as.numeric(as.character(validation$ech))
validation <- validation[,-c(5,28)]


library(caret)

# Fitting model

fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)

fit7 <- train(tH2_obs ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)

predicted= predict(fit7,validation) 
err7 <- sqrt(mean((predicted-validation[,1])**2))

#########
# autre gbm
#########

train <- train[,-c(3,13)]
validation <- validation[,-c(3,13)]

tH2_obs = train$tH2_obs
train = select(train,-tH2_obs)
end_trn = nrow(train)


all = rbind(train,validation[,-1])
end = nrow(all)

names(all)
all = select(all,capeinsSOL0,clwcH20,ddH10_rose4,ffH10,flir1SOL0,
             fllat1SOL0,flsen1SOL0,flvis1SOL0,hcoulimSOL0,huH2,
             nbSOL0_HMoy,nH20,ntSOL0_HMoy,pMER0,rrH20,tH2,
             tH2_VGrad_2.100,tH2_XGrad,tH2_YGrad,tpwHPA850,ux1H10,vapcSOL0,
             vx1H10,ech,mois,jour  
)

ntrees = 5000

model7 = gbm.fit(
  x = all[1:end_trn,],
  y = tH2_obs,
  distribution="gaussian",
  n.trees = ntrees,
  shrinkage = 0.1,
  interaction.depth = 6,
  n.minobsinnode = 10,
  nTrain = round(end_trn * 0.8),
  verbose = TRUE
  
)
predicted= predict(object = model7,newdata = validation[,-1],
                   n.trees = gbm.perf(model7, plot.it = FALSE)
) 

err7 <- sqrt(mean((predicted-validation[,1])**2))

vect_err <- c(err1,err2,err3,err4,err5,err6,err7)
mean(vect_err)

### SEPARATION DE TEST PAR VILLES

for (city in villes){
  fichier <- subset(testnew,insee == city)
  assign(paste(city,"_","test",sep = ""),fichier[,-2])
  
}

###########
# Ville 1 #
###########

prediction1<-predict(reg1,newdata=`6088001_test`)
t1<-c(prediction1,`6088001_test`)
`6088001_test`$tH2predict<-prediction1
`6088001_test`$insee <- 6088001

#### ModÃ¨le boost

`6088001_test`$ech<-as.numeric(as.character(`6088001_test`$ech))

prediction1<-predict(fit1,newdata=`6088001_test`)
t1<-c(prediction1,`6088001_test`)
`6088001_test`$tH2predict<-prediction1
`6088001_test`$insee <- 6088001

### autre modÃ¨le

predicted1= predict(object = model1,newdata = `6088001_test`,
                   n.trees = gbm.perf(model1, plot.it = FALSE)
) 
`6088001_test`$tH2predict<-predicted1
`6088001_test`$insee <- 6088001

###########
# Ville 2 #
###########

prediction2<-predict(reg2,newdata=`31069001_test`)
t2<-c(prediction2,`31069001_test`)
`31069001_test`$tH2predict<-prediction2
`31069001_test`$insee <- 31069001

#### ModÃ¨le boost

`31069001_test`$ech<-as.numeric(as.character(`31069001_test`$ech))

prediction2<-predict(fit2,newdata=`31069001_test`)
t2<-c(prediction2,`31069001_test`)
`31069001_test`$tH2predict<-prediction2
`31069001_test`$insee <- 31069001

### autre modÃ¨le

predicted2= predict(object = model2,newdata = `31069001_test`,
                    n.trees = gbm.perf(model2, plot.it = FALSE)
) 
`31069001_test`$tH2predict<-predicted2
`31069001_test`$insee <- 31069001

###########
# Ville 3 #
###########

prediction3<-predict(reg3,newdata=`33281001_test`)
t3<-c(prediction3,`33281001_test`)
`33281001_test`$tH2predict<-prediction3
`33281001_test`$insee <- 33281001

#### ModÃ¨le boost

`33281001_test`$ech<-as.numeric(as.character(`33281001_test`$ech))

prediction3<-predict(fit3,newdata=`33281001_test`)
t3<-c(prediction3,`33281001_test`)
`33281001_test`$tH2predict<-prediction3
`33281001_test`$insee <- 33281001

### autre modÃ¨le

predicted3= predict(object = model3,newdata = `33281001_test`,
                    n.trees = gbm.perf(model3, plot.it = FALSE)
) 
`33281001_test`$tH2predict<-predicted3
`33281001_test`$insee <- 33281001

###########
# Ville 4 #
###########

prediction4<-predict(reg4,newdata=`35281001_test`)
t4<-c(prediction4,`35281001_test`)
`35281001_test`$tH2predict<-prediction4
`35281001_test`$insee <- 35281001

#### ModÃ¨le boost

`35281001_test`$ech<-as.numeric(as.character(`35281001_test`$ech))

prediction4<-predict(fit4,newdata=`35281001_test`)
t4<-c(prediction4,`35281001_test`)
`35281001_test`$tH2predict<-prediction4
`35281001_test`$insee <- 35281001

### autre modÃ¨le

predicted4= predict(object = model4,newdata = `35281001_test`,
                    n.trees = gbm.perf(model4, plot.it = FALSE)
) 
`35281001_test`$tH2predict<-predicted4
`35281001_test`$insee <- 35281001

###########
# Ville 5 #
###########

prediction5<-predict(reg5,newdata=`59343001_test`)
t5<-c(prediction5,`59343001_test`)
`59343001_test`$tH2predict<-prediction5
`59343001_test`$insee <- 59343001

#### ModÃ¨le boost

`59343001_test`$ech<-as.numeric(as.character(`59343001_test`$ech))

prediction5<-predict(fit5,newdata=`59343001_test`)
t5<-c(prediction5,`59343001_test`)
`59343001_test`$tH2predict<-prediction5
`59343001_test`$insee <- 59343001

### autre modÃ¨le

predicted5= predict(object = model5,newdata = `59343001_test`,
                    n.trees = gbm.perf(model5, plot.it = FALSE)
) 
`59343001_test`$tH2predict<-predicted5
`59343001_test`$insee <- 59343001

###########
# Ville 6 #
###########

prediction6<-predict(reg6,newdata=`67124001_test`)
t6<-c(prediction6,`67124001_test`)
`67124001_test`$tH2predict<-prediction6
`67124001_test`$insee <- 67124001

#### ModÃ¨le boost

`67124001_test`$ech<-as.numeric(as.character(`67124001_test`$ech))

prediction6<-predict(fit6,newdata=`67124001_test`)
t6<-c(prediction6,`67124001_test`)
`67124001_test`$tH2predict<-prediction6
`67124001_test`$insee <- 67124001

### autre modÃ¨le

predicted6= predict(object = model6,newdata = `67124001_test`,
                    n.trees = gbm.perf(model6, plot.it = FALSE)
) 
`67124001_test`$tH2predict<-predicted6
`67124001_test`$insee <- 67124001

###########
# Ville 7 #
###########

prediction7<-predict(reg7,newdata=`75114001_test`)
t7<-c(prediction7,`75114001_test`)
`75114001_test`$tH2predict<-prediction7
`75114001_test`$insee <- 75114001

#### ModÃ¨le boost

`75114001_test`$ech<-as.numeric(as.character(`75114001_test`$ech))

prediction7<-predict(fit7,newdata=`75114001_test`)
t7<-c(prediction7,`75114001_test`)
`75114001_test`$tH2predict<-prediction7
`75114001_test`$insee <- 75114001

### autre modÃ¨le

predicted7= predict(object = model7,newdata = `75114001_test`,
                    n.trees = gbm.perf(model7, plot.it = FALSE)
) 
`75114001_test`$tH2predict<-predicted7
`75114001_test`$insee <- 75114001




answer <- rbind(`6088001_test`,`31069001_test`,`33281001_test`,`35281001_test`,`59343001_test`,`67124001_test`,`75114001_test`)

answer$ech<-as.numeric(as.character(answer$ech))
ordonne <- answer[order(answer$ech,answer$date),]

sapply(ordonne,class)
write.table(ordonne[,c(1,32,24,31)],"gbm.csv",row.names = FALSE,sep=";")

test_answer<-read.csv2("test_answer_template.csv",header=TRUE,sep=";")

## Tester que c'est bien ordonnÃ©e
sqrt(mean((test$tH2-ordonne$tH2predict)**2))
any(is.na(ordonne$tH2predict))

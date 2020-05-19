#2020 Final

rm(list = ls())    #delete objects

crime = read.csv("~/2020crime.txt", header=FALSE)

cat("\014")
library(imputeTS)
library(glmnet)
library(rmutil)
library(tictoc)
library(latex2exp)
library(class) # KNN 
library(ggplot2) # A library to generate high-quality graphs
library(dplyr) # A library for data manipulation
library(gridExtra) # Provides a number of user-level functions to work with "grid" graphics, notably to arrange multiple grid-based plots on a page, and draw tables
library(MASS)
library(caret)
library(randomForest)
library(readxl)
library(e1071)
library(logistf)

#Get the data


MyData        =  crime
MyData        =  MyData[,-c(1:5)]
MyData.length = ncol(MyData)

MyData = na_mean(MyData)

n = nrow(MyData)
p = ncol(MyData)

names(MyData)[p] = "crime_rate"

#Create the matrices to store the results
result_test = data.frame(matrix(nrow = 100, ncol = 4))
colnames(result_test) <- c("RF", "Elastic-net", "Lasso", "Ridge")

result_train = data.frame(matrix(nrow = 100, ncol = 4))
colnames(result_train) <- c("RF", "Elastic-net", "Lasso", "Ridge")

for (i in 1:100) {
  
  #set random split for each 100 iterations (will be a different random split for each i)
  set.seed(i)
  
  
  samplesize      = floor(.8*n) #change to .9 for part 2
  trainindex      = sample(seq_len(n), size = samplesize) 
  D_Learn         = MyData[trainindex,]
  D_Validate      = MyData[-trainindex,]

  ## Lasso and Ridge ##
  
  x_learn      = model.matrix(crime_rate~.,D_Learn)[,-MyData.length]
  y_learn      = D_Learn$crime_rate
  x_validate   = model.matrix(crime_rate~.,D_Validate)[,-MyData.length]
  y_validate   = D_Validate$crime_rate
  
  lasso.cv = cv.glmnet(x_learn, y_learn, nfolds = 10, 
                       alpha = 1, intercept = FALSE)
  
  lasso.model = glmnet(x_learn, y_learn, 
                          alpha = 1, lambda = lasso.cv$lambda.best, intercept = FALSE)
  
  ridge.cv = cv.glmnet(x_learn, y_learn, nfolds = 10, 
                       alpha = 0, intercept = FALSE)
  ridge.model = glmnet(x_learn, y_learn, 
                          alpha = 0, lambda = ridge.cv$lambda.best, intercept = FALSE)
  
  enet.cv = cv.glmnet(x_learn, y_learn, nfolds = 10, 
                       alpha = .5, intercept = FALSE)
  enet.model = glmnet(x_learn, y_learn, 
                          alpha = .5, lambda = enet.cv$lambda.best, intercept = FALSE)
  
  rf.model = randomForest(x_learn, y_learn,mtry = sqrt(p), importance = TRUE)

  lasso.train.hat  =     predict(lasso.model, newx = x_learn, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  lasso.test.hat   =     predict(lasso.model, newx = x_validate, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  lasso.train.err  =     mean((lasso.train.hat - y_learn)^2)
  lasso.test.err   =     mean((lasso.test.hat - y_validate)^2)
  rsqtrain.lasso   =     1 - (lasso.train.err/(lasso.train.err+lasso.test.err))
  rsqtest.lasso    =     1 - (lasso.test.err/(lasso.train.err+lasso.test.err))
  
  ridge.train.hat  =     predict(ridge.model, newx = x_learn, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  ridge.test.hat   =     predict(ridge.model, newx = x_validate, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  ridge.train.err  =     mean((ridge.train.hat - y_learn)^2)
  ridge.test.err   =     mean((ridge.test.hat - y_validate)^2)
  rsqtrain.ridge   =     1 - (ridge.train.err/(ridge.train.err+ridge.test.err))
  rsqtest.ridge   =      1 - (ridge.test.err/(ridge.train.err+ridge.test.err))
  
  enet.train.hat  =     predict(enet.model, newx = x_learn, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  enet.test.hat   =     predict(enet.model, newx = x_validate, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  enet.train.err  =     mean((enet.train.hat - y_learn)^2)
  enet.test.err   =     mean((enet.test.hat - y_validate)^2)
  rsqtrain.enet   =     1 - (enet.train.err/(enet.train.err+enet.test.err))
  rsqtest.enet    =     1 - (enet.test.err/(enet.train.err+enet.test.err))
  
  rf.train.hat  =     predict(rf.model, x_learn) # y.train.hat=X.train %*% fit$beta + fit$a0
  rf.test.hat   =     predict(rf.model, x_validate) # y.test.hat=X.test %*% fit$beta  + fit$a0
  rf.train.err  =     mean((rf.train.hat - y_learn)^2)
  rf.test.err   =     mean((rf.test.hat - y_validate)^2)
  rsqtrain.rf   =     1 - (rf.train.err/(rf.train.err+rf.test.err))
  rsqtest.rf    =     1 - (rf.test.err/(rf.train.err+rf.test.err))
   
  #### RESULTS FROM HW****
  result_train[i, 1] <- rsqtrain.rf
  result_train[i, 2] <- rsqtrain.enet
  result_train[i, 3] <- rsqtrain.lasso
  result_train[i, 4] <- rsqtrain.ridge
  
  result_test[i, 1] <- rsqtest.rf
  result_test[i, 2] <- rsqtest.enet
  result_test[i, 3] <- rsqtest.lasso
  result_test[i, 4] <- rsqtest.ridge
}

###Plot from HW#####


#Plot the results for test error
boxplot(result_test,
        main = "R-Squared(Test) for 100 Randomly Chosen Training Sets",
        xlab = "Model Building Method",
        ylab = "R-Squared",
        las = 1,
        col = c("lightsteelblue","lightskyblue", "steelblue", "slateblue"),
        names = c("RF", "E-net", "Lasso", "Ridge")
)
#Plot the results for training error
boxplot(result_train,
        main = "R-Squared(Train) for 100 Randomly Chosen Training Sets",
        xlab = "Model Building Method",
        ylab = "R-Squared",
        las = 1,
        col = c("lightsteelblue","lightskyblue", "steelblue", "slateblue"),
        names = c("RF", "E-net", "Lasso", "Ridge")
)

plot(lasso.cv)
plot(ridge.cv)
plot(enet.cv)

#Plot the residuals for training data
boxplot(rf.train.hat, enet.train.hat, lasso.train.hat, ridge.train.hat,
        main = "Residuals for Training Data",
        xlab = "Model Building Method",
        ylab = "Residuals",
        las = 1,
        col = c("lightsteelblue","lightskyblue", "steelblue", "slateblue"),
        names = c("RF", "E-net", "Lasso", "Ridge")
)
#Plot the residuals for test data
boxplot(rf.test.hat, enet.test.hat, lasso.test.hat, ridge.test.hat,
        main = "Residuals for Test Data",
        xlab = "Model Building Method",
        ylab = "Residuals",
        las = 1,
        col = c("lightsteelblue","lightskyblue", "steelblue", "slateblue"),
        names = c("RF", "E-net", "Lasso", "Ridge")
)


plot(lasso.model)
plot(ridge.model)
plot(enet.model)
plot(rf.model)

lasso_coef = predict(lasso.model, type = "coefficients")
lasso_coef[lasso_coef != 0]

ridge_coef = predict(ridge.model, type = "coefficients")
ridge_coef[lasso_coef != 0]

enet_coef = predict(enet.model, type = "coefficients", s= enet.cv$lambda)
enet_coef[lasso_coef != 0]

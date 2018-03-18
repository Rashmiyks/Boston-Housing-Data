#Loading packages
library(MASS)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(corrplot)
library(leaps)
library(rpart)
library(tree)
library(mgcv)
library(glmnet)
library(boot)
library(rpart.plot)
library(nnet)

#Exploratory Data Analysis
data(Boston)
dim(Boston)
head(Boston)
glimpse(Boston)
summary(Boston)

#Check for missing values
sum(is.na(Boston))

#Check for duplicated values
sum(duplicated(Boston))

#checking correlation between variables
corrplot(cor(Boston), method = "number", type = "upper", diag = FALSE) 

#Visualization
Boston %>%
  gather(key, val, -chas) %>%
  ggplot(aes(x = val, fill = as.factor(chas))) +
  geom_histogram(bins = 30) +
  facet_wrap(~key, scales = "free") +
  theme_gray() +
  ggtitle("Distribution of variables in Boston Housing Data")

#scatter plots
Boston %>%
  gather(key, val, -medv) %>%
  ggplot(aes(x = val, y = medv)) +
  geom_point() +
  stat_smooth(method = "lm", se = TRUE, col = "blue") +
  facet_wrap(~key, scales = "free") +
  theme_gray() +
  ggtitle("Scatter plot of predictor variables vs Median Value (medv)") 

#Count of values in chas variable
table(Boston$chas)

#Split into 80:20 training and testing data set
set.seed(12383010)
index <- sample(nrow(Boston), nrow(Boston) * 0.80)
Boston.train <- Boston[index, ]
Boston.test <- Boston[-index, ]

#fitting a linear regression model
model1 <- lm(medv ~ ., data = Boston.train)
model1.sum <- summary(model1)
model1.sum

#Looking at model summary, we see that variables indus and age are insignificant
#Building model without variables indus and age
model2 <- lm(medv ~ . -indus -age, data = Boston.train)
model2.sum <- summary(model2)
model2.sum

#Model Diagnostics for model 2
par(mfrow = c(2,2))
plot(model2)
par(mfrow = c(1,1))

#Variable Selection using best subset regression
model.subset <- regsubsets(medv ~ ., data = Boston.train, nbest = 1, nvmax = 13)
summary(model.subset)
plot(model.subset, scale = "bic")

#Variable selection using stepwise regression
nullmodel <- lm(medv ~ 1, data = Boston.train)
fullmodel <- lm(medv ~ ., data = Boston.train)

#forward selection
model.step.f <- step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), direction = "forward")

#Backward selection
model.step.b <- step(fullmodel, direction = "backward")

#stepwise selection
model.step <- step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), direction = "both")
AIC(model.step)
BIC(model.step)
summary(model.step)

#In-sample performance
#MSE
model1.sum <- summary(model1)
(model1.sum$sigma) ^ 2

model2.sum <- summary(model2)
(model2.sum$sigma) ^ 2

#R-squared
model1.sum$r.squared
model2.sum$r.squared

#Adjusted r square
model1.sum$adj.r.squared
model2.sum$adj.r.squared

#AIC 
AIC(model1)
AIC(model2)

#BIC
BIC(model1)
BIC(model2)

#Out-of-sample Prediction or test error (MSPE)
model1.pred.test <- predict(model1, newdata = Boston.test)
model1.mspe <- mean((model1.pred.test - Boston.test$medv) ^ 2)
model1.mspe

model2.pred.test <- predict(model2, newdata = Boston.test)
model2.mspe <- mean((model2.pred.test - Boston.test$medv) ^ 2)
model2.mspe

#Cross Validation
model1.glm = glm(medv ~ ., data = Boston)
cv.glm(data = Boston, glmfit = model1.glm, K = 5)$delta[2]

model2.glm <- glm(medv ~ . -indus -age, data = Boston)
cv.glm(data = Boston, glmfit = model2.glm, K = 5)$delta[2]

#########################################################

#Fitting a linear regression tree
Boston.tree <- rpart(medv ~ ., data = Boston.train)
Boston.tree

#Plotting the tree
rpart.plot(Boston.tree, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE, extra = 1)

plotcp(Boston.tree)
printcp(Boston.tree)

#Building a large tree
Boston.largetree <- rpart(formula = medv ~ ., data = Boston.train, cp = 0.001)
plot(Boston.largetree)
text(Boston.largetree)

printcp(Boston.largetree)
plotcp(Boston.largetree)

#however, from plotcp, we observe that a tree with more than 7 to 9 splits is not very helpful.
#further pruning the tree to limit to 9 splits;corresponding cp value from plot is 0.0072
pruned.tree <- prune(Boston.largetree, cp = 0.0072)
pruned.tree
rpart.plot(pruned.tree, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE, extra = 1)

#In-sample MSE
mean((predict(Boston.tree) - Boston.train$medv) ^ 2)      #default tree
mean((predict(Boston.largetree) - Boston.train$medv) ^ 2)  #large tree
mean((predict(pruned.tree) - Boston.train$medv) ^ 2)       #pruned tree

#out-of-sample performance
#Mean squared error loss for this tree
mean((predict(Boston.tree, newdata = Boston.test) - Boston.test$medv) ^ 2)  #default tree
mean((predict(Boston.largetree, newdata = Boston.test) - Boston.test$medv) ^ 2)   #large tree
mean((predict(pruned.tree, newdata = Boston.test) - Boston.test$medv) ^ 2)     #pruned tree

############################################################################

#Generalized Additive Model (GAM)
Boston.gam <- gam(medv ~ s(crim) + s(zn) + s(indus) + s(nox) + s(rm) + s(age) + s(dis) + 
                    s(tax) + s(ptratio) + s(black) + s(lstat) + chas + rad, data = Boston.train)
summary(Boston.gam)

#Model AIC, BIC, mean residual deviance
AIC(Boston.gam)
BIC(Boston.gam)
Boston.gam$deviance

#plot
plot(Boston.gam, shade = TRUE, seWithMean = TRUE, scale = 0)

#In-sample prediction
(Boston.gam.mse <- mean((predict(Boston.gam) - Boston.train$medv) ^ 2))

#Out-of-sample prediction - MSPE
(Boston.gam.mspe <- mean((predict(Boston.gam, newdata = Boston.test) - Boston.test$medv) ^ 2))




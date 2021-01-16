
##### Entire Project R Code #####
# Gunjan Neopaney
# Erik Solis
# Edward Bernal
# Gundeep Singh

##### Beginning of KNN Model Process done by Gunjan Neopaney and Edward Bernal #####
setwd("~/Desktop/University of Houston Classes/Spring 2020/Data Science and Stats Learning/Project/Final Items")
grad_admission <- read.csv("Admission_Predict.csv")

grad_admission$Serial.No. <- NULL
grad_admission$Research <- as.factor(grad_admission$Research)

# Normalizing the Chance of Admit values so the minimum value is 0 and maximum value is 1.
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
norm_Chance.of.Admit <- normalize(grad_admission$Chance.of.Admit)
grad_admission <- cbind(grad_admission,norm_Chance.of.Admit)

## Setting up the classes for response variable using quantiles.
# Class 3 = “Highly Likely” -> x >= 0.78
# Class 2 = “Likely”        -> 0.78 > x >= 0.62
# Class 1 = “Unlikely”      -> 0.62 > x >= 0.48
# Class 0 = “Highly Unlikely” -> x < 0.48

grad_admission$Class_chance <- ifelse(norm_Chance.of.Admit>=quantile(norm_Chance.of.Admit,.75),
                                      3,norm_Chance.of.Admit)
grad_admission$Class_chance <- ifelse(quantile(norm_Chance.of.Admit,.75)>norm_Chance.of.Admit & norm_Chance.of.Admit >= quantile(norm_Chance.of.Admit,.50),
                                      2,grad_admission$Class_chance)
grad_admission$Class_chance <- ifelse(quantile(norm_Chance.of.Admit,.50)>norm_Chance.of.Admit & norm_Chance.of.Admit >= quantile(norm_Chance.of.Admit,.25),
                                      1,grad_admission$Class_chance)
grad_admission$Class_chance <- ifelse(norm_Chance.of.Admit<quantile(norm_Chance.of.Admit,.25),
                                      0,grad_admission$Class_chance)
grad_admission$Class_chance <- as.factor(grad_admission$Class_chance)

# Creating exploratory data analysis plots
par(mfrow=c(2,3))
for (j in 1:6) plot(grad_admission[,j] ~ factor(grad_admission$Class_chance),
                    ylab=colnames(grad_admission)[j],
                    xlab="Class_chance")
par(mfrow=c(1,1))

# Creating and scaling training and testing sets.
n <- nrow(grad_admission)
set.seed(2)
train <- sample(1:n, n*.8)
X.train <- scale(grad_admission[train,c(1:6)])
y.train <- grad_admission[train, 'Class_chance']
X.test <- scale(grad_admission[-train,c(1:6)],
                center = attr(X.train, "scaled:center"),
                scale = attr(X.train, "scaled:scale"))
y.test <- grad_admission[-train, 'Class_chance']

### Building the KNN model
library(class)
# Picking multiple k values from 1 to 100 seperated by 4.
K.set <- seq(1,100, by = 4)
knn.test.err <- numeric(length(K.set))

set.seed(2)
# KNN prediction model that runs on all the k values mentioned above.
for (j in 1:length(K.set)){
  knn.pred <- knn(train = X.train, test = X.test,
                  cl= y.train,
                  k=K.set[j])
  knn.test.err[j] <- mean(knn.pred != y.test)}
min(knn.test.err) #Prints minimum test error.
which.min(knn.test.err) #Prints the index of the k value.
K.set[which.min(knn.test.err)] #Prints the actual k value of the minimum error.
plot(K.set, knn.test.err, type='b', main = 'KNN Test Error vs Corresponding K values')


### Fitting whole data set on our best model (KNN, with K = 57)
set.seed(2)
knn.pred <- knn(train=scale(grad_admission[,1:6]), test=scale(grad_admission[,1:6]),
                cl=grad_admission[,10],
                k=57)
mean(knn.pred != grad_admission[,10])
summary(knn.pred)
table(knn.pred, grad_admission[,10])



##### Beginning of SVM Model Process done by Erik Solis and Gundeep Singh #####


##### Preliminary Stuff #####

grad_admission <- read.csv("~/Data Science 2/Project/graduate-admissions/Admission_Predict.csv")

grad_admission$Serial.No. <- NULL

str(grad_admission)  #To check the classes for each preditors.
grad_admission$Research <- as.factor(grad_admission$Research)



##### Normalizing the Chance.of.Admit values so min is 0 and max is 1 #####

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
grad_admission$norm_Chance.of.Admit <- normalize(grad_admission$Chance.of.Admit)

attach(grad_admission)



##### Setting up the classes for response variable #####
#Class 3 = 'Highly Likely' -> norm_Chance.of.Admit >= 0.78
#Class 2 = 'Likely'        -> 0.78 > norm_Chance.of.Admit >= 0.62
#Class 1 = 'Unlikely'      -> 0.62 > norm_Chance.of.Admit >= 0.48
#Class 0 = 'Highly Unlikely' -> norm_Chance.of.Admit < 0.48
#####
grad_admission$class_chance <- ifelse(norm_Chance.of.Admit>=quantile(norm_Chance.of.Admit,.75),
                                      3,norm_Chance.of.Admit)
grad_admission$class_chance <- ifelse(quantile(norm_Chance.of.Admit,.75)>norm_Chance.of.Admit &
                                        norm_Chance.of.Admit >= quantile(norm_Chance.of.Admit,.50),
                                      2,grad_admission$class_chance)
grad_admission$class_chance <- ifelse(quantile(norm_Chance.of.Admit,.50)>norm_Chance.of.Admit &
                                        norm_Chance.of.Admit >= quantile(norm_Chance.of.Admit,.25),
                                      1,grad_admission$class_chance)
grad_admission$class_chance <- ifelse(norm_Chance.of.Admit<quantile(norm_Chance.of.Admit,.25),
                                      0,grad_admission$class_chance)

grad_admission$class_chance <- as.factor(grad_admission$class_chance)

grad_admission$Chance.of.Admit <- NULL         # Remove original Chance.of.Admit
grad_admission$norm_Chance.of.Admit <- NULL    # Remove normalized Chance.of.Admit
grad_admission$Research <- NULL                # Remove Research variable
attach(grad_admission)



##### SVM Model Creation #####

library(e1071)

# Data partitioning
RNGkind(sample.kind = 'Rounding')
set.seed(2)
n = nrow(grad_admission)
tsamp = sample(1:n, 0.8*n)

train = grad_admission[tsamp,]
test = grad_admission[-tsamp,]

# support vector classifier
set.seed(2)
tune.linear=tune(method=svm,
                 class_chance~.,data=train,
                 kernel="linear",
                 ranges=list(cost=c(0.001,0.1,1,5,10,100)))
summary(tune.linear)
plot(tune.linear$best.model,data=train,GRE.Score~TOEFL.Score)

# Train/Test error for support vector classifier
mean(predict(tune.linear$best.model) != class_chance[tsamp])
mean(predict(tune.linear$best.model, newdata=test) != class_chance[-tsamp])


# polynomial svm
set.seed(2)
tune.poly=tune(svm,class_chance~., data=train,
               kernel="polynomial",
               ranges=list(cost=c(0.001,0.1,1,5,10,100),
                           degree=c(2,3,4,5)))
summary(tune.poly)
#plot(tune.poly$best.model,data=train,GRE.Score~CGPA)

# Train/Test error for polynomial SVM
mean(predict(tune.poly$best.model) != class_chance[tsamp])
mean(predict(tune.poly$best.model, newdata=test) != class_chance[-tsamp])


# radial svm
set.seed(2)
tune.rad=tune(svm,class_chance~., data=train,
              kernel="radial",
              ranges=list(cost=c(0.001,0.1,1,5,10,100),
                          gamma=c(0.5,1,2,3,4)))
summary(tune.rad)
#plot(tune.rad$best.model,data=train,GRE.Score~University.Rating)

# Train/Test error for radial SVM
mean(predict(tune.rad$best.model) != class_chance[tsamp])
mean(predict(tune.rad$best.model, newdata=test) != class_chance[-tsamp])



##### Misc Plots #####

library(ggplot2)
# ggplot of class_chance by GRE.Score v. CGPA
e <- ggplot(grad_admission,aes(CGPA,GRE.Score))
e + geom_point(aes(color=class_chance)) + 
  theme(legend.position='right') + labs(title='Chance of Admission by GRE Score and CGPA')

library(ggplot2)
# ggplot of class_chance by GRE.Score v.TOEFL.Score 
e <- ggplot(grad_admission,aes(TOEFL.Score,GRE.Score))
e + geom_point(aes(color=class_chance)) + 
  theme(legend.position='right') + labs(title='Chance of Admission by GRE and TOEFL Scores')









# Load dataset
library(dslabs)
mnist <- read_mnist()
train_images <- mnist$train$images
test_images <- mnist$test$images
train_labels <- mnist$train$labels
test_labels <- mnist$test$labels


# Inspect contents
#summary(train_images)
summary(train_labels)

# How the picture looks like
train_images[1,]
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
show_digit(train_images[1,])


# Normalize the numeric data
train_normalized <- as.data.frame(scale(train_images,scale = FALSE,center = TRUE))
test_normalized <- as.data.frame(scale(test_images,scale = FALSE,center = TRUE))



# PCA
set.seed(132)
# pca_train <- prcomp(digits_covMatrix)
pca_train <- prcomp(train_normalized)


# The percent of the variances in data
variance_explained <- as.data.frame(pca_train$sdev^2/sum(pca_train$sdev^2)) 
variance_explained <- cbind(c(1:784), cumsum(variance_explained)) 
colnames(variance_explained) <- c("NmbrPCs","CumVar") 
# Look at 50th variance
variance_explained[50, ]



# Plot the data.
par(mfrow=c(2,2))
plot (variance_explained$NmbrPCs, variance_explained$CumVar, 
      xlab = "Number of Factors", ylab = "Proportion of Variance Explained", 
      type = "l", col = "red")
plot (pca_train$sdev^2/sum(pca_train$sdev^2), xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", type = "b" )
plot(pca_train, type = "l", main = "Scree plot")
plot(pca_train, type = "barplot", main = "Scree plot")


# summary(pca_train)
summary(pca_train)$importance[,1:50]


# Do the actual dimension reduction (matrix of 784 columns should be converted 
# to matrix of 50 columns)
pca_rot <- pca_train$rotation[,1:50]
train_final <- as.matrix(train_images)%*%(pca_rot)  
test_final <- as.matrix(test_images)%*%(pca_rot) 
test_final <- data.frame(test_final)
train_final <- data.frame(train_final)



# Visualize the different classes in two dimensions
library(ggplot2)
ggplot(train_final, aes(PC1,PC2,color=train_labels)) + geom_point() +
  labs(x= "PC1",y= "PC2") + ggtitle("PCA") 


# Inspect the reconstruction of the original data from these two components
mean.train = colMeans(train_images)
PCA <- c(1,10,50,784)
par(mfrow=c(2,2))
i <- 1
while(i <= length(PCA))
{  
  nComp = PCA[i]
  train.pca.re = pca_train$x[,1:nComp] %*% t(pca_train$rotation[,1:nComp])
  train.pca.re = scale(train.pca.re, center = -mean.train,scale=F)
  show_digit(train.pca.re[1,])
  i<-i+1
}

# So PCA = 50 is acceptable to see the images of the digits. 



# Run KNN with k from 1 to 20
library(class)
set.seed(132)
knn_acc <- matrix(nrow=20,ncol=2)
i=1
for (k in 1:20){
  prediction = knn(train_final,test_final,train_labels,k=k)
  knn_acc[i,] <- c(k,mean(prediction==test_labels))
  i = i+1
}
t(knn_acc)


# Plot the accuracy for different k (1:20)
par(mfrow=c(1,1))
plot(knn_acc[,1],knn_acc[,2],type = "l",
     xlab = "Value of k", 
     ylab = "Testing Accuracy",
     main = "Choosing value of k based on testing accuracy")


# From the plot, choose the k = 8, which gives the best accuracy.
best_prediction <- knn(train_final,test_final,train_labels,k=8)
test_labels_knn <- factor(test_labels)


# Confusion Matrix
library(caret)
confusionMatrix(best_prediction, test_labels_knn)


# Calculate the accuracy for k = 8
hit = 0 
for (i in 1:10){
  hit = hit + table(best_prediction,test_labels)[i,i]
}
table_prediction <- table(prediction,test_labels)
table_prediction
accuracy = hit/sum(table_prediction)
accuracy


# Random Forest with PCA
library(randomForest)
library(readr)
library(lattice)
set.seed(132) 
numTrain <- 40000 
numTrees <- 50

# Generate a random sample of "numTrain" indexes
rows <- sample(1:nrow(train_final), numTrain) 
train_labels <- factor(train_labels)
rf <- randomForest(train_final, train_labels, ntree=numTrees)
rf
plot(rf)

# Make prediction
pred <- predict(rf, test_final)
test_labels_rf <- factor(test_labels)

# Confusion Matrix with PCA
confusionMatrix(pred, test_labels_rf)



# Run Random Forest without PCA
train_final2 <- as.matrix(train_images)
test_final2 <- as.matrix(test_images)
test_final2 <- data.frame(test_final2)
train_final2 <- data.frame(train_final2)

rows <- sample(1:nrow(train_final2), numTrain) 
train_labels <- factor(train_labels)
rf <- randomForest(train_final2, train_labels, ntree=numTrees)
rf
plot(rf)

# Make prediction
pred2 <- predict(rf, test_final2)
test_labels_rf2 <- factor(test_labels)

# Confusion Matrix without PCA
confusionMatrix(pred2, test_labels_rf2)




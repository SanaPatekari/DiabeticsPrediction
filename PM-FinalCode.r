## The data that we are working with is the Women in Data Science(WiDS) 2021 Datathon data.
## The task is to determine whether a patient admitted to an ICU has been diagnosed with a particular type of diabetes i.e Diabetes Mellitus.
## The problem is a binary classification proble# Load necessary libraries
library(dplyr)
library(VIM)
library(DataExplorer)
library(mice)
library(caret)
library(corrplot)
library(e1071)
library(ggplot2)
library(pROC)
library(caret)
library(ggplot2)
library(reshape2)

#Load the data 
data <- read.csv("C:/Fall 2024/MA5790/PM-Project/widsdatathon2021/TrainingWiDS2021.csv")
summary(data)
head(data)

#Lets check basic info about the data
ncol(data)
nrow(data)
colnames(data)
data$diabetes_mellitus

# Calculate proportions for the target variable
target_proportions <- table(data$diabetes_mellitus)
target_df <- as.data.frame(target_proportions)
target_df
colnames(target_df) <- c("Class", "Count")
print(target_df)

# Create the pie chart
ggplot(target_df, aes(x = "", y = Count, fill = factor(Class))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar("y") +
  labs(title = "Proportion of Class 0 and 1", fill = "Class") +
  theme_void() +
  geom_text(aes(label = scales::percent(Count/sum(Count), accuracy = 0.1)), position = position_stack(vjust = 0.5))


## We can see that we have 180 predictors and 130157 observations
## Lets keep diabetes_mellitus in a separate variable
target_col <- data$diabetes_mellitus
target_col_name <- "diabetes_mellitus"
target_col

# Use data explorer to see data
#create_report(data)

data <- data[ , !names(data) %in% "diabetes_mellitus"]
data

# See unique values in columns
sapply(data, function(x) length(unique(x)))

## Two variables X and encounter_id only contains unique values so we can drop them
## The variable hospital_id doesn't provide any useful information. 
## It is not a relevant feature whether a patient has been diagnosed with diabetes. It might be used to partition data later.
## Lets drop all these columns from our predictor

data <- data %>% select(-X, -encounter_id, -hospital_id)

# Calculate the missing data for each column 
missing_summary <- colSums(is.na(data))
missing_summary_df <- data.frame(Variable = names(missing_summary),Missing_Count = missing_summary,Missing_Percentage = (missing_summary / nrow(data)) * 100)
print(missing_summary_df)

## Calculate total rows with missing observations
sum(rowSums(is.na(data)) > 0)

# Identify columns with high missing values  > 70%
high_missing_df <- missing_summary_df[missing_summary_df$Missing_Percentage > 70, ]
cat("Columns with missing values > 70%")
print(high_missing_df$Variable)

## Remove columns with >70% missing values
data <- data[, !(names(data) %in% high_missing_df$Variable)]
ncol(data)

## Separate categorical and numerical columns
categorical_cols <- c('ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source','icu_stay_type', 'icu_type')
numerical_cols <- setdiff(colnames(data), c(categorical_cols, target_col))
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

## Use mice imputation to fill the missing values
mice_imputed <- mice(data, m = 3, method = c('pmm'), maxit = 3)
data_mice_imputed <- complete(mice_imputed)

length(data_mice_imputed)

## One-hot encode categorical columns
cols <- as.formula(paste("~", paste(categorical_cols, collapse = "+")))
dummy_vars <- dummyVars(cols, data = data_mice_imputed, fullRank = TRUE)
dummy_data <- predict(dummy_vars, newdata = data_mice_imputed)
dummy_data <- data.frame(dummy_data)
data_mice_imputed <- data_mice_imputed[, !(names(data_mice_imputed) %in% categorical_cols)]
data_encoded <- cbind(data_mice_imputed, dummy_data)
head(data_encoded)

## See about data after imputation
create_report(data_encoded)
data_encoded <- data_encoded %>% select(-h1_inr_max, -h1_inr_min)#Still has misssing values
create_report(data_encoded)

#Find out which variables are near zero variance
nzv <- nearZeroVar(data_encoded, saveMetrics = TRUE)
nzv_summary <- nzv[nzv$nzv == TRUE | nzv$zeroVar == TRUE, ]
if (nrow(nzv_summary) > 0) {
  cat("Zero & Near Zero Variance Predictors:\n")
  print(nzv_summary)
}
# Remove near-zero variance predictors from the data
cleaned_data <- data_encoded[, !nzv$nzv]
str(cleaned_data)

#saving cleaned data before removing correlated predictors for models which can handle high coliearity
GPrep_data<-cleaned_data
length(GPrep_data)

#Calculate the correlation matrix
cor_matrix <- cor(cleaned_data)
corrplot(cor_matrix, method = "color")

#Find high correlation columns
highCorr <- findCorrelation(cor_matrix, cutoff = 0.8) 
length(highCorr)
highCorr
cleaned_data <- cleaned_data[, -highCorr]
length(cleaned_data)

#Correlation plot after removing predictors 
cor_matrix_clean <- cor(cleaned_data)
corrplot(cor_matrix_clean, method = "color", order="hclust")
#create_report(cleaned_data)

nrow(cleaned_data) #Number of observations
ncol(cleaned_data) #No of predictors

# Calculate skewness for each predictor
skewness_values <- apply(cleaned_data, 2, skewness)
skewness_df <- data.frame(Feature = names(skewness_values), Skewness = skewness_values)
skewness_df_sorted <- skewness_df[order(skewness_df$Skewness), ]
print(skewness_df_sorted)

# Select columns that are suitable for Yeo-Johnson transformation
high_skew_cols <- skewness_df$Feature[abs(skewness_df$Skewness) > 0.75] 

# Look at the distribution before applying Yeo-Johnson
par(mfrow = c(10, 7))
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(cleaned_data)) {
  # Plot the histogram with normalized probabilities
  hist(cleaned_data[[colname]], 
       main = paste("Histogram of", colname), 
       xlab = colname, 
       border = "black", 
       probability = TRUE)
  lines(density(cleaned_data[[colname]]), col = "red", lwd = 1)
}
par(mfrow = c(1, 1))

# Apply Yeo-Johnson transformation 
preProcYeoJohnson <- preProcess(cleaned_data[, high_skew_cols], method = c("YeoJohnson"))
print(preProcYeoJohnson)

# Apply the transformation
yeoJohnson_data <- cleaned_data
yeoJohnson_data[, high_skew_cols] <- predict(preProcYeoJohnson, cleaned_data[, high_skew_cols])

# Plot the data after transformation
par(mfrow = c(10, 7))
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(yeoJohnson_data)) {
  # Plot the histogram with normalized probabilities
  hist(yeoJohnson_data[[colname]], 
       main = paste("Histogram of", colname), 
       xlab = colname, 
       border = "black", 
       probability = TRUE)
  lines(density(yeoJohnson_data[[colname]]), col = "red", lwd = 1)
}
par(mfrow = c(1, 1))

# Check skewness after transformation
skewTransformed <- apply(yeoJohnson_data, 2, skewness)
skewness_trsfm_df <- data.frame(Feature = names(skewTransformed), Skewness = skewTransformed)
skewness_trsfm_sorted <- skewness_trsfm_df[order(skewness_trsfm_df$Skewness), ]
print(skewness_trsfm_sorted)


#Use boxplots to visualize 
par(mfrow = c(10, 7))  
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(yeoJohnson_data)) {
  boxplot(yeoJohnson_data[[colname]], 
          main = paste("Boxplot of", colname), 
          xlab = colname, 
          col = "lightblue", 
          border = "black")
}
par(mfrow = c(1, 1))

##Apply Spatial Sign
preProcSpatial <- preProcess(yeoJohnson_data, method = "spatialSign")
trsfmd_data <- predict(preProcSpatial, yeoJohnson_data)
head(trsfmd_data)

#Use boxplots to visualize after transformation
par(mfrow = c(10, 7))  
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(trsfmd_data)) {
  boxplot(trsfmd_data[[colname]], 
          main = paste("Boxplot of", colname), 
          xlab = colname, 
          col = "lightblue", 
          border = "black")
}
par(mfrow = c(1, 1))
final_data <- data.frame(trsfmd_data, class = target_col) ##Add class to the data

colnames(final_data)
nrow(final_data)

## Create another with PCA 
pca_data_scaled <- scale(trsfmd_data)
pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

# Plot the variance explained by each principal component
plot(pca_result, type = "l", main = "Scree Plot")
pca_df <- data.frame(pca_result$x, class = final_data$class)
# Visualize the first two principal components
ggplot(pca_df, aes(x = PC1, y = PC2, color = class)) +
  geom_point(size = 2) +
  labs(title = "PCA: First Two Principal Components",
       x = "PC1",
       y = "PC2") +
  theme_minimal()

table(final_data$Class)
head(final_data)

###Data Spliting
set.seed(123) 
trainIndex <- createDataPartition(final_data$class, p = 0.8, list = FALSE)  
trainData <- final_data[trainIndex, ]
testData <- final_data[-trainIndex, ]

## Create train and test
X_train <- trainData[, setdiff(names(trainData), "class")]
y_train <- factor(trainData$class, levels = c(0, 1), labels = c("No_Diabetes", "Diabetes"))

X_test <- testData[, setdiff(names(testData), "class")]
y_test <- factor(testData$class, levels = c(0, 1), labels = c("No_Diabetes", "Diabetes"))

trCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = defaultSummary, savePredictions = "final",verboseIter = TRUE)

# Initialize a list to store model results
results <- list()
#lda model
lda_model <- train(x = X_train, y = y_train,method = "lda",  preProcess = c("center", "scale"),  trControl = trCtrl,  metric = "Kappa")
results[[model_name]] <- lda_model
lda_preds_test <- predict(lda_model, X_test)
lda_conf_test <- confusionMatrix(lda_preds_test, y_test)

## Train the NNNet Model
nnModel <- train(X_train, y_train, method = "nnet",trControl = trCtrl, tuneLength = 5, preProcess = c("center", "scale"), metric = "Kappa")
results[[model_name]] <- nnModel
plot(nnModel)
nnPred <- predict(nnModel, X_test)
nn_conf_matrix <- confusionMatrix(data = nnPred, reference = y_test)

## SVM Model
svmModel <- train(X_train, y_train, method = 'svmRadial',trControl = trCtrl, tuneLength = 5, preProcess = c("center", "scale"), metric = "Kappa")
results[[model_name]] <- svmModel
plot(svmModel)
svmPred <- predict(svmModel, X_test)
confusionMatrix(data = svmPred, reference = y_test)

## KNN Model
knnModel <- train(X_train, y_train, method = 'knn',trControl = trCtrl, tuneLength = 5, preProcess = c("center", "scale"), metric = "Kappa")
results[[model_name]] <- knnModel
plot(knnModel)
knnPred <- predict(knnModel, X_test)
confusionMatrix(data = knnPred, reference = y_test)

## Naive Bayes Model
nbModel <- train(X_train, y_train, method = 'naive_bayes',trControl = trCtrl, tuneLength = 5, preProcess = c("center", "scale"), metric = "Kappa")
results[[model_name]] <- nbModel
nbPred <- predict(nbModel, X_test)
confusionMatrix(data = nbPred, reference = y_test)

###Model without removing highly correlated
#seperate preprocess and data split
str(GPrep_data)
nrow(GPrep_data) 
ncol(GPrep_data) 

# Calculate skewness for each predictor
skewness_values_G <- apply(GPrep_data, 2, skewness)
skewness_df_G <- data.frame(Feature = names(skewness_values_G), Skewness = skewness_values_G)
skewness_df_sorted_G <- skewness_df_G[order(skewness_df_G$Skewness), ]
print(skewness_df_sorted_G)

# Select columns that are suitable for Yeo-Johnson transformation
high_skew_cols_G <- skewness_df$Feature[abs(skewness_df$Skewness) > 0.75] 
high_skew_cols_G
# Look at the distribution before applying Yeo-Johnson
par(mfrow = c(10, 7))
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(GPrep_data)) {
  # Plot the histogram with normalized probabilities
  hist(GPrep_data[[colname]], 
       main = paste("Histogram of", colname), 
       xlab = colname, 
       border = "black", 
       probability = TRUE)
  lines(density(GPrep_data[[colname]]), col = "red", lwd = 1)
}
par(mfrow = c(1, 1))artial

# Apply Yeo-Johnson transformation 
preProcYeoJohnson <- preProcess(GPrep_data[, high_skew_cols_G], method = c("YeoJohnson"))
print(preProcYeoJohnson)

# Apply the transformation
yeoJohnson_data_G <- GPrep_data
yeoJohnson_data_G[, high_skew_cols_G] <- predict(preProcYeoJohnson, GPrep_data[, high_skew_cols_G])

# Plot the data after transformation
par(mfrow = c(10, 7))
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(yeoJohnson_data_G)) {
  # Plot the histogram with normalized probabilities
  hist(yeoJohnson_data_G[[colname]], 
       main = paste("Histogram of", colname), 
       xlab = colname, 
       border = "black", 
       probability = TRUE)
  lines(density(yeoJohnson_data_G[[colname]]), col = "red", lwd = 1)
}
par(mfrow = c(1, 1))

# Check skewness after transformation
skewTransformed_1 <- apply(yeoJohnson_data_G, 2, skewness)
skewness_trsfm_df_1 <- data.frame(Feature = names(skewTransformed_1), Skewness = skewTransformed_1)
skewness_trsfm_sorted_1 <- skewness_trsfm_df_1[order(skewness_trsfm_df_1$Skewness), ]
print(skewness_trsfm_sorted_1)


#Use boxplots to visualize 
par(mfrow = c(10, 7))  
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(yeoJohnson_data_G)) {
  boxplot(yeoJohnson_data_G[[colname]], 
          main = paste("Boxplot of", colname), 
          xlab = colname, 
          col = "lightblue", 
          border = "black")
}
par(mfrow = c(1, 1))

##Apply Spatial Sign
preProcSpatial <- preProcess(yeoJohnson_data_G, method = "spatialSign")
trsfmd_data_G <- predict(preProcSpatial, yeoJohnson_data_G)
head(trsfmd_data_G)

#Use boxplots to visualize after transformation
par(mfrow = c(10, 7))  
par(mar = c(1, 1, 1, 1)) 
for (colname in colnames(trsfmd_data_G)) {
  boxplot(trsfmd_data_G[[colname]], 
          main = paste("Boxplot of", colname), 
          xlab = colname, 
          col = "lightblue", 
          border = "black")
}
par(mfrow = c(1, 1))
final_data_G <- data.frame(trsfmd_data_G, class = target_col) ##Add class to the data

colnames(final_data_G)
nrow(final_data_G)
str(final_data_G)

#data split
set.seed(123) 
trainIndex_G <- createDataPartition(final_data_G$class, p = 0.8, list = FALSE)  
trainData_G <- final_data_G[trainIndex_G, ]
testData_G <- final_data_G[-trainIndex_G, ]

## Create train and test
X_train_G <- trainData_G[, setdiff(names(trainData_G), "class")]
y_train_G <- factor(trainData_G$class, levels = c(0, 1), labels = c("No_Diabetes", "Diabetes"))

X_test_G <- testData_G[, setdiff(names(testData_G), "class")]
y_test_G <- factor(testData_G$class, levels = c(0, 1), labels = c("No_Diabetes", "Diabetes"))

#Logistic model
logistic_model <- train( x = X_train_G, y = y_train_G, method = "glm", family = "binomial",preProcess = c("center", "scale"),  trControl = trCtrl,metric = "Kappa")
results[[model_name]] <- logistic_model
logistic_preds_test <- predict(logistic_model, X_train_G)
logistic_conf_test <- confusionMatrix(logistic_preds_test, y_train_G)
logistic_conf_test 

#plsda model
plsda_model <- train(x = X_train_G, y = y_train_G,method = "pls", tuneGrid = expand.grid(.ncomp = 1:30), preProcess = c("center", "scale"), trControl = trCtrl, metric = "Kappa")
results[[model_name]] <- plsda_model
plot(plsda_model)
plsda_preds_test <- predict(plsda_model, X_train_G)
plsda_conf_test <- confusionMatrix(plsda_preds_test, y_train_G)
plsda_conf_test 

#tunegrid
tune_grid <- expand.grid(alpha = 1,lambda = seq(0.001, 0.1, by = 0.001))

#penalized model
penalized_model_tune <- train(x = X_train_G, y = y_train_G,method = "glmnet",  family = "binomial",  preProcess = c("center", "scale"),  trControl = trCtrl,  tuneGrid = tune_grid,  metric = "Kappa")
results[[model_name]] <- penalized_model_tune
plot(penalized_model_tune)
penalized_preds_test <- predict(penalized_model_tune, X_train_G)
penalized_conf_test <- confusionMatrix(penalized_preds_test, y_train_G)

# MDA
set.seed(123)
mda_model <- train(x = X_train_G,y = y_train_G,method = "mda",preProcess = c("center", "scale"),trControl = trCtrl,metric = "Kappa",tuneGrid = expand.grid(subclasses = 1:10)  )
results[[model_name]] <- mda_model
plot(mda_model)
mda_preds_test <- predict(mda_model, X_train_G)
mda_conf_test <- confusionMatrix(mda_preds_test, y_train_G)

#FDA
set.seed(123)
fda_model <- train(x = X_train_G, y = y_train_G,method = "fda",  preProcess = c("center", "scale"),trControl = trCtrl,tuneLength = 10,#tuneGrid = marsGrid,metric = "Kappa")
results[[model_name]] <- fda_model
plot(fda_model)
fda_preds_test <- predict(fda_model, X_train_G)
fda_conf_test <- confusionMatrix(fda_preds_test, y_train_G)

# QDA 
qda_model <- train(x = X_train, y = y_train,method = "qda", preProcess = c("center", "scale"),trControl = trCtrl,tuneLength = 10,metric = "Kappa")
results[[model_name]] <- qda_model
qda_preds_test <- predict(qda_model, X_train_G)
qda_conf_test <- confusionMatrix(qda_preds_test, y_train_G)

#Confusion matrix plot for top 2 models
plot_confusion_matrix <- function(cm) {
  cm_matrix <- as.table(cm)
  
  cm_melted <- melt(cm_matrix)
  colnames(cm_melted) <- c("Reference", "Prediction", "Count")
  
  ggplot(cm_melted, aes(x = Prediction, y = Reference, fill = Count)) +
    geom_tile(color = "white", size = 0.5) +  
    geom_text(aes(label = Count), color = "black", size = 6) +  
    scale_fill_gradient(low = "#cfe2f3", high = "#1f4e79") +  
    theme_minimal() +
    labs(
      title = "Confusion Matrix (Test Set)", 
      x = "Predicted", 
      y = "Actual"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12),  
      axis.text.y = element_text(size = 12),  
      axis.title = element_text(size = 14),
      plot.title = element_text(size = 16, hjust = 0.5)  
    )
}

plot_confusion_matrix(mda_conf_test)
plot_confusion_matrix(nn_conf_matrix)

#VarImp for optimal model
mda_var_importance <- varImp(mda_model, scale = TRUE)
print(mda_var_importance)
print(mda_var_importance, top=10)
plot(mda_var_importance, top=10)
# -----------------------------------------------------------------------------
# This code snippet was generated for educational purposes as part of the
# Intelligent Systems course at the University of Deusto. The code has been
# created with assistance from ChatGPT version 3.5 and GitHub Copilot.
#
# The code is released under the Creative Commons License and is provided
# for free use and modification by the programming and development community.
#
# This script was generated in April 2024, the year when the Athletic Club de
# Bilbao won the 25th King's Cup.
# -----------------------------------------------------------------------------

# Install required packages
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)

# Clear plots
if(!is.null(dev.list())) dev.off()
# Clear console
cat("\014")
# Clean workspace
rm(list=ls())
# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load custom functions
source("decision-tree-learning-utils.R")

#-----------------------
# LOAD AND PREPARE DATA
#-----------------------
# Read data from CSV
filename = "../data/2022_car.csv"
data <- read.csv(file = filename, sep =",", header = TRUE)

#------------------------
# PLOT DATA DISTRIBUTION
#------------------------
# Folder to save images
images_folder <- "../data/images/2022-car-seller-type/"
# Plot the relation between the target variable and the rest of the columns
plot.data.distribution(data, target = "seller_type", folder=images_folder)

#----------------------------------------
# TRANSFORM NUMERIC COLUMNS INTO FACTORS
#----------------------------------------
# YEAR
index1 <- data$year <= 2005
index2 <- data$year > 2005 & data$year <= 2010
index3 <- data$year > 2010 & data$year <= 2015
index4 <- data$year > 2015
data$year[index1] <- "<= 2005"
data$year[index2] <- "2006 - 2010"
data$year[index3] <- "2011 - 2015"
data$year[index4] <- "> 2015"

# SELLING PRICE
data$selling_price <- as.numeric(data$selling_price / 100)
index1 <- data$selling_price > 10000
index2 <- data$selling_price >= 7500 & data$selling_price <= 10000
index3 <- data$selling_price >= 5000 & data$selling_price < 7500
index4 <- data$selling_price < 5000 & data$selling_price >= 2000
index5 <- data$selling_price < 2500
data$selling_price[index1] <- "> 10,000"
data$selling_price[index2] <- "7,500 - 10,000"
data$selling_price[index3] <- "5,000 - 7,499"
data$selling_price[index4] <- "2,500 - 4,999"
data$selling_price[index5] <- "< 2,500"

# KM DRIVEN
index1 <- data$km_driven > 100000
index2 <- data$km_driven >= 75000 & data$km_driven <= 100000
index3 <- data$km_driven >= 50000 & data$km_driven < 75000
index4 <- data$km_driven >= 25000 & data$km_driven < 50000
index5 <- data$km_driven < 25000
data$km_driven[index1] <- "> 100,000"
data$km_driven[index2] <- "75,000 - 100.000"
data$km_driven[index3] <- "50.000 - 74,999"
data$km_driven[index4] <- "25,000 - 49,999"
data$km_driven[index5] <- "< 25,000"

# FUEL
data <- subset(data, fuel != as.factor("Electric"))

data$name = NULL

#---------------------
# PLOT DATA FREQUENCY
#---------------------
# Plot the frequency of each attribute
plot.data.frequency(data, folder=images_folder)

#---------------------------------------------------
# GENERATE AND ANALYZE DECISION TREE LEARNING MODEL
#---------------------------------------------------
# Percentaje os training examples
training_p <- 0.75

best_model <- NA
best_accuracy <- 0

for(i in 1:10) {
  # Generate data partition 75% training / 25% test. The result is a vector with
  # the indexes of the examples that will be used for the training of the model.
  training_indexes <- createDataPartition(y = data$seller_type, p = training_p, list = FALSE)

  # Split training and test data
  training_data <- data[training_indexes, ]  # Extract training data using training_indexes
  test_data     <- data[-training_indexes, ] # Extract data with the indexes not included in training_indexes

  # Create Linear Model using training data. Formula = all the columns except Salary
  model <- rpart(formula = seller_type~., data = training_data)

  # Make the prediction using the model and test data
  prediction <- predict(model, test_data, type = "class")

  # Calculate accuracy using Confusion Matrix
  prediction_results <- table(test_data$seller_type, prediction)
  matrix <- confusionMatrix(prediction_results)
  accuracy <- matrix$overall[1]
  # Get attribute names order by importance
  attr <- names(model$variable.importance)

  # Print accuracy and top 5 attributes
  cat(paste0(i, ".- Accuracy = ", round(100*accuracy, digits = 2)),
      "%. Top 5 attributes:\n")
  # Print top 5 attributes
  for (j in 1:5) {
    print(paste0("    Attribute-", j, " = ", attr[j]), quote = FALSE)
  }

  # Update best model if accuracy is better
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_model <- model
  }
}

# Print accurcy and attributes in order of relevance
attrs <- names(best_model$variable.importance)
cat(paste0("Accuracy = ", round(100*best_accuracy, digits = 2)),
    "%\nAttributes ordered by relevance:", paste0("\n  ", attrs))

# Plot tree (this method is slow, wait until pot is completed)
rpart.plot(best_model,
           type = 2,
           extra = 102,
           tweak = 1.1,
           box.palette = "BuGn",
           shadow.col = "darkgray",
           main = "Car Seller Type Decision Tree",
           sub = paste0("Accuracy = ", round(100*best_accuracy, digits = 2), "%"))

# Print the rules that represent the Decision Tree
rpart.rules(best_model,
            style="tall",
            trace = 0,
            cover = TRUE,
            eq = "==",
            when = "IF",
            and = "&&",
            extra = 4)

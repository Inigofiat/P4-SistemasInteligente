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
filename = "../data/2023_heart-disease.csv"
data <- read.csv(file = filename, sep =",", header = TRUE)

# Transform HeartDisease into NO and YES
data$HeartDisease <- ifelse(data$HeartDisease == 0, "NO (0)", "YES (1)")

#------------------------
# PLOT DATA DISTRIBUTION
#------------------------
# Folder to save images
images_folder <- "../data/images/2023-heart-disease/"
# Plot the relation between the target variable and the rest of the columns
plot.data.distribution(data, target = "HeartDisease", folder=images_folder)

#----------------------------------------
# TRANSFORM NUMERIC COLUMNS INTO FACTORS
#----------------------------------------
# Create 4 bims for Age (based on lowest and highest values)
bin_width <- round((max(data$Age) - min(data$Age)) / 4)
labels <- paste0(min(data$Age) + (1:4 - 1) * bin_width, " <-> ", min(data$Age) + 1:4 * bin_width)
data$Age <- cut(data$Age, breaks = 4, labels = labels)
# Create 3 bims for RestingBP (Normal, High, Very High)
data$RestingBP <- ifelse(data$RestingBP < 100, "Normal (<100)",
                         ifelse(data$RestingBP <= 150, "High (>=100 && <=150)", "Very High (>150)"))
# Create 3 bims for MaxHR (Low, Normal, High)
data$MaxHR <- ifelse(data$MaxHR < 150, "Low (<=150)",
                     ifelse(data$MaxHR <= 180, "Normal (>=150 && <=180)", "High (>180)"))
# Create two values for Cholesterol (Normal, High, Very High)
data$Cholesterol <- ifelse(data$Cholesterol <= 200, "Normal (<=200)",
                           ifelse(data$Cholesterol < 300, "High (>=200 && <=300)", "Very High (>300)"))
# Transform FastingBS into NO and YES
data$FastingBS <- ifelse(data$FastingBS == 0, "NO (0)", "YES (1)")
# Transform ExerciseAngina into NO and YES
data$ExerciseAngina <- ifelse(data$ExerciseAngina == "N", "NO", "YES")
# Create bims for Oldpeak Define labels based on lowest and highest value of each bin
data$Oldpeak <- ifelse(data$Oldpeak < 0, "Negative (<0)",
                       ifelse(data$Oldpeak <= 2, "Normal-1 (>=0 && <=2)",
                              ifelse(data$Oldpeak <= 2.5, "Normal-2 (>=2.0 && <=2.5)", "High (>2.5)")))
#---------------------
# PLOT DATA FREQUENCY
#---------------------
# Plot the frequency of each attribute
plot.data.frequency(data, folder=images_folder)

#---------------------------------------------------
# GENERATE AND ANALYZE DECISION TREE LEARNING MODEL
#---------------------------------------------------
# Percentage of training examples
training_p <- 0.75

best_model <- NA
best_accuracy <- 0

for(i in 1:10) {
  # Generate data partition 75% training / 25% test. The result is a vector with
  # the indexes of the examples that will be used for the training of the model.
  training_indexes <- createDataPartition(y = data$HeartDisease, p = training_p, list = FALSE)

  # Split training and test data
  training_data <- data[training_indexes, ]
  test_data     <- data[-training_indexes, ]

  # Create Linear Model using training data. Formula = all the columns except Salary
  model <- rpart(formula = HeartDisease~., data = training_data)

  # Make the prediction using the model and test data
  prediction <- predict(model, test_data, type = "class")

  # Calculate accuracy using Confusion Matrix
  prediction_results <- table(test_data$HeartDisease, prediction)
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
           main = "Heart Disease Decision Tree",
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

library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(reshape2)

rm(list=ls())
cat("\014")
if(!is.null(dev.list())) dev.off()
graphics.off()

data <- read.csv("data/2025_loan_approval_dataset.csv")

str(data)
summary(data)

valoresFaltantes <- colSums(is.na(data))
print("Valores faltantes por columna:")
print(valoresFaltantes)

varNum <- sapply(data, is.numeric)
columnasNum <- names(data)[varNum]
print("Variables numéricas en el dataset:")
print(columnasNum)

summary(data[, columnasNum])

relacionNum <- function(data, var_name) {
  p <- ggplot(data, aes(x = .data[[var_name]], y = loan_status)) +
    geom_point(alpha = 0.5) +
    labs(title = paste("Relación entre", var_name, "y aprobación de préstamo"),
         x = var_name, y = "Estado del préstamo") +
    theme_minimal()

  if(var_name == "income_annum") {
    y_pos <- as.character(levels(data$loan_status)[1])
    p <- p + annotate("text",
                      x = max(data[[var_name]], na.rm = TRUE) * 0.7,
                      y = y_pos,
                      label = "Este gráfico nos puede ayudar a\nestablacer rangos adecuados",
                      color = "darkred", size = 3.5, hjust = 0)
  }

  print(p)
  return(p)
}

varImp <- c("income_annum", "loan_amount", "loan_term", "cibil_score")
plots <- list()

for(var in varImp) {
  dev.new()
  plots[[var]] <- relacionNum(data, var)
}

cat("\nCreando rangos para income_annum con umbrales constantes:\n")
data$income_range <- ifelse(data$income_annum <= 2500000, "Bajo", 
                            ifelse(data$income_annum <= 5000000, "Medio", "Alto"))

cat("\nCreando rangos para cibil_score con umbrales constantes:\n")
data$cibil_range <- ifelse(data$cibil_score <= 600, "Riesgo Alto",
                           ifelse(data$cibil_score <= 750, "Riesgo Medio", "Riesgo Bajo"))

print("Distribución de rangos de ingresos:")
print(table(data$income_range))

print("Distribución de rangos de puntuación crediticia:")
print(table(data$cibil_range))

print("Relación entre rangos de ingresos y aprobación de préstamos:")
print(table(data$income_range, data$loan_status))

print("Relación entre rangos de puntuación crediticia y aprobación de préstamos:")
print(table(data$cibil_range, data$loan_status))

cat("\nCreando rangos uniformes para loan_amount:\n")
data$loan_amount_range <- cut(data$loan_amount, breaks = 4)

cat("\nCreando rangos uniformes para loan_term:\n")
data$loan_term_range <- cut(data$loan_term, breaks = 3)

print("Distribución de rangos de monto de préstamo:")
print(table(data$loan_amount_range))

print("Distribución de rangos de plazo de préstamo:")
print(table(data$loan_term_range))

print("Relación entre rangos de monto de préstamo y aprobación:")
print(table(data$loan_amount_range, data$loan_status))

print("Relación entre rangos de plazo de préstamo y aprobación:")
print(table(data$loan_term_range, data$loan_status))

ingresosPrestamos <- as.data.frame(prop.table(table(data$income_range, data$loan_status), margin = 1) * 100)
colnames(ingresosPrestamos) <- c("Income_Range", "Loan_Status", "Percentage")

dev.new()
ggplot(ingresosPrestamos, aes(x = Income_Range, y = Percentage, fill = Loan_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)),
            position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Porcentaje de préstamos aprobados por rango de ingresos",
       x = "Rango de ingresos", y = "Porcentaje") +
  theme_minimal()

pCredPrestamos <- as.data.frame(prop.table(table(data$cibil_range, data$loan_status), margin = 1) * 100)
colnames(pCredPrestamos) <- c("Cibil_Range", "Loan_Status", "Percentage")

dev.new()
ggplot(pCredPrestamos, aes(x = Cibil_Range, y = Percentage, fill = Loan_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Porcentaje de préstamos aprobados por rango de puntuación crediticia",
       x = "Rango de puntuación crediticia", y = "Porcentaje") +
  theme_minimal()

for(col in colnames(data)) {
  if(sum(is.na(data[[col]])) > 0) {
    if(is.numeric(data[[col]])) {
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
      data[[col]][is.na(data[[col]])] <- mode_val
    }
  }
}

varCategoricas <- c("education", "self_employed", "loan_status")
for(var in varCategoricas) {
  if(var %in% colnames(data)) {
    data[[var]] <- as.factor(data[[var]])
  }
}

print("Distribución de aprobación de préstamos:")
table(data$loan_status)

print("Relación entre educación y aprobación de préstamos:")
table(data$education, data$loan_status)

print("Relación entre autoempleo y aprobación de préstamos:")
table(data$self_employed, data$loan_status)

varNum <- c("income_annum", "loan_amount", "loan_term", "cibil_score")

dev.new()
par(mfrow=c(2,2))
for(var in varNum) {
  if(var %in% colnames(data)) {
    hist(data[[var]], main=paste("Histograma de", var), xlab=var, col="lightblue")
  }
}
par(mfrow=c(1,1))

varOrig <- function() {
  set.seed(123)
  indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  dataPrueba <- data[indexPrueba, ]
  dataTest <- data[-indexPrueba, ]
  
  formulaOrig <- loan_status ~ income_annum + loan_amount + loan_term + cibil_score + 
    education + self_employed + no_of_dependents
  
  modeloOrig <- rpart(formulaOrig, data = dataPrueba, method = "class", 
                      control = rpart.control(maxdepth = 5))
  
  predOrig <- predict(modeloOrig, dataTest, type = "class")
  confMatrixOrig <- confusionMatrix(predOrig, dataTest$loan_status)
  
  return(list(
    accuracy = confMatrixOrig$overall["Accuracy"],
    model = modeloOrig
  ))
}

pruebaRangos <- function() {
  set.seed(123)
  indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  dataPrueba <- data[indexPrueba, ]
  dataTest <- data[-indexPrueba, ]
  
  formulaRangos <- loan_status ~ income_range + cibil_range + loan_amount_range + 
    loan_term_range + education + self_employed + no_of_dependents
  
  modeloRangos <- rpart(formulaRangos, data = dataPrueba, method = "class", 
                        control = rpart.control(maxdepth = 5))
  
  predRangos <- predict(modeloRangos, dataTest, type = "class")
  confMatrixRangos <- confusionMatrix(predRangos, dataTest$loan_status)
  
  return(list(
    accuracy = confMatrixRangos$overall["Accuracy"],
    model = modeloRangos
  ))
}

cat("\nEvaluando rendimiento de modelos con variables originales vs rangos...\n")
resultadosOrig <- varOrig()
resultadosRangos <- pruebaRangos()

cat("\nResultados de evaluación:\n")
cat("Precisión modelo con variables originales:", round(resultadosOrig$accuracy, 4), "\n")
cat("Precisión modelo con variables por rangos:", round(resultadosRangos$accuracy, 4), "\n")

set.seed(123)

indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
dataPrueba <- data[indexPrueba, ]
dataTest <- data[-indexPrueba, ]

arbolPrueba <- function(dataPrueba, dataTest, max_depth = 5, min_split = 20, cp = 0.01) {
  modeloArbol <- rpart(loan_status ~ ., 
                       data = dataPrueba, 
                       method = "class",
                       control = rpart.control(maxdepth = max_depth, 
                                               minsplit = min_split,
                                               cp = cp))
    predicciones <- predict(modeloArbol, dataTest, type = "class")
    matrixConf <- confusionMatrix(predicciones, dataTest$loan_status)
    accuracy <- matrixConf$overall["Accuracy"]
    
  if("Pos Pred Value" %in% names(matrixConf$byClass)) {
    precisionAprobada <- matrixConf$byClass["Pos Pred Value"]
  } else {
    precisionAprobada <- NA
  }
  
  if("Neg Pred Value" %in% names(matrixConf$byClass)) {
    precisionDenegada <- matrixConf$byClass["Neg Pred Value"]
  } else {
    precisionDenegada <- NA
  }
    return(list(
    model = modeloArbol,
    predicciones = predicciones,
    matrixConf = matrixConf,
    accuracy = accuracy,
    precisionAprobada = precisionAprobada,
    precisionDenegada = precisionDenegada
  ))
}

resultados <- list()

configuraciones <- list(
  list(max_depth = 3, min_split = 10, cp = 0.01),
  list(max_depth = 4, min_split = 15, cp = 0.01),
  list(max_depth = 5, min_split = 20, cp = 0.01),
  list(max_depth = 6, min_split = 25, cp = 0.01),
  list(max_depth = 4, min_split = 20, cp = 0.02),
  list(max_depth = 5, min_split = 25, cp = 0.015),
  list(max_depth = 6, min_split = 20, cp = 0.005),
  list(max_depth = 7, min_split = 15, cp = 0.008),
  list(max_depth = 5, min_split = 30, cp = 0.012),
  list(max_depth = 4, min_split = 25, cp = 0.018)
)

set.seed(123)
for(i in 1:10) {
  cat("Entrenando árbol", i, "...\n")
  
  set.seed(i * 100)
  
  muestra_indices <- sample(1:nrow(dataPrueba), size = floor(0.8 * nrow(dataPrueba)))
  muestra_train <- dataPrueba[muestra_indices, ]
  resultados[[i]] <- arbolPrueba(
    muestra_train, 
    dataTest,
    max_depth = configuraciones[[i]]$max_depth,
    min_split = configuraciones[[i]]$min_split,
    cp = configuraciones[[i]]$cp
  )
}

accuracies <- sapply(resultados, function(r) r$accuracy)
print("Precisiones de los 10 árboles:")
print(round(accuracies, 4))

mejorModeloIndex <- which.max(accuracies)
mejorModelo <- resultados[[mejorModeloIndex]]

dev.new()
print(paste("El mejor árbol es el #", mejorModeloIndex, "con precisión:", round(mejorModelo$accuracy, 4)))
rpart.plot(mejorModelo$model, extra = 104, box.palette = "RdBu", shadow.col = "gray")

reglas <- rpart.rules(mejorModelo$model, roundint = FALSE)
print("Reglas que definen el comportamiento del árbol:")
print(reglas)

set.seed(123)
modeloRf <- randomForest(loan_status ~ ., data = dataPrueba)
varImportante <- importance(modeloRf)
varImportanteDf <- data.frame(
  Variable = row.names(varImportante),
  Importance = varImportante[,1]
)
varImportanteDf <- varImportanteDf[order(varImportanteDf$Importance, decreasing = TRUE),]

print("Los 5 atributos más relevantes:")
print(head(varImportanteDf, 5))

dev.new()
barplot(varImportanteDf$Importance[1:5], 
        names.arg = varImportanteDf$Variable[1:5],
        col = "steelblue",
        main = "Importancia de Variables",
        ylab = "Importancia",
        las = 2)

educationAccuracy <- mean(mejorModelo$predicciones[dataTest$education == " Graduate"] == 
                            dataTest$loan_status[dataTest$education == " Graduate"])
print(paste("Precisión considerando solo educación (Graduate):", round(educationAccuracy, 4)))

accuracyTrabajadorAuto <- mean(mejorModelo$predicciones[dataTest$self_employed == " Yes"] == 
                                 dataTest$loan_status[dataTest$self_employed == " Yes"])
print(paste("Precisión considerando solo trabajadores autónomos:", round(accuracyTrabajadorAuto, 4)))

resultadosTabla <- data.frame(
  Tree = 1:10,
  Accuracy = sapply(resultados, function(r) r$accuracy),
  PrecisionApproved = sapply(resultados, function(r) r$precisionAprobada),
  PrecisionRejected = sapply(resultados, function(r) r$precisionDenegada)
)

resultadosTabla <- rbind(resultadosTabla, 
                         data.frame(Tree = "Media", 
                                    Accuracy = mean(resultadosTabla$Accuracy),
                                    PrecisionApproved = mean(resultadosTabla$PrecisionApproved, na.rm = TRUE),
                                    PrecisionRejected = mean(resultadosTabla$PrecisionRejected, na.rm = TRUE)))

print("Tabla de resultados de los 10 árboles:")
print(resultadosTabla, digits = 4)

dev.new()

resultadosLong <- melt(resultadosTabla[1:10,], id.vars = "Tree", 
                       variable.name = "Metric", value.name = "Value")

ggplot(resultadosLong, aes(x = factor(Tree), y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(data = resultadosTabla[11,], 
             aes(yintercept = Accuracy, color = "Media Accuracy"), 
             linetype = "dashed") +
  geom_hline(data = resultadosTabla[11,], 
             aes(yintercept = PrecisionApproved, color = "Media Precision Approved"), 
             linetype = "dashed") +
  geom_hline(data = resultadosTabla[11,], 
             aes(yintercept = PrecisionRejected, color = "Media Precision Rejected"), 
             linetype = "dashed") +
  labs(title = "Resultados de los 10 árboles de decisión",
       x = "Árbol", y = "Valor") +
  theme_minimal()

cat("\nCONCLUSIONES DEL ANÁLISIS DE RANGOS:\n")
cat("1. Se analizaron las variables numéricas del dataset (income_annum, loan_amount, loan_term, cibil_score).\n")
cat("2. Se crearon rangos usando umbrales constantes para income_annum y cibil_score.\n")
cat("3. Se crearon rangos uniformes para loan_amount y loan_term.\n")

if(resultadosRangos$accuracy > resultadosOrig$accuracy) {
  cat("4. El modelo con variables categorizadas en rangos mejora la precisión en ", 
      round((resultadosRangos$accuracy - resultadosOrig$accuracy) * 100, 2), 
      "% respecto al modelo con variables originales.\n")
  cat("5. La transformación de variables numéricas a rangos parece beneficiar el rendimiento del modelo.\n")
} else {
  cat("4. El modelo con variables originales tiene mejor precisión que el modelo con rangos (diferencia de ", 
      round((resultadosOrig$accuracy - resultadosRangos$accuracy) * 100, 2), "%).\n")
  cat("5. La transformación de variables numéricas a rangos no parece beneficiar el rendimiento del modelo en este caso.\n")
}

cat("6. Los rangos creados ofrecen mayor interpretabilidad a costa de posible pérdida de información detallada.\n")

cat("\nCONCLUSIONES GENERALES DEL MODELADO:\n")
cat("- Se han generado 10 árboles de decisión con diferentes configuraciones para predecir la aprobación de préstamos.\n")
cat(paste("- La precisión media de los modelos es del:", round(mean(accuracies) * 100, 2), "%\n"))
cat(paste("- El mejor modelo (árbol #", mejorModeloIndex, ") tiene una precisión del:", 
          round(mejorModelo$accuracy * 100, 2), "%\n"))
cat("- Las variables más relevantes para la predicción son:\n")
for(i in 1:5) {
  cat(paste("  ", i, ". ", varImportanteDf$Variable[i], "\n"))
}

mejorArbol <- mejorModelo$model


dataTestPredic <- dataTest
dataTestPredic$predicted_status <- mejorModelo$predicciones

analisisDependiente <- dataTest %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    actual_approved = sum(loan_status == " Approved"),
    actual_approved_pct = actual_approved / total * 100
  )

prediccionesDependientes <- dataTestPredic %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    predicted_approved = sum(predicted_status == " Approved"),
    predicted_approved_pct = predicted_approved / total * 100
  )

resultadosDependientes <- merge(analisisDependiente, prediccionesDependientes, by="no_of_dependents")
resultadosDependientes <- resultadosDependientes %>%
  select(no_of_dependents, actual_approved_pct, predicted_approved_pct)

longDependientes <- melt(resultadosDependientes, 
                         id.vars = "no_of_dependents",
                         variable.name = "status_type", 
                         value.name = "percentage")

dev.new()
ggplot(longDependientes, aes(x=factor(no_of_dependents), y=percentage, fill=status_type)) +
  geom_bar(stat="identity", position="dodge") +
  geom_text(aes(label=sprintf("%.1f%%", percentage)), 
            position=position_dodge(width=0.9), vjust=-0.3) +
  labs(title="Porcentaje de préstamos aprobados por número de dependientes",
       subtitle="Valores reales vs predichos",
       x="Número de dependientes",
       y="Porcentaje de aprobación") +
  scale_fill_manual(values=c("actual_approved_pct"="steelblue", 
                             "predicted_approved_pct"="coral"),
                    labels=c("Real", "Predicho")) +
  theme_minimal() +
  theme(legend.title = element_blank())

clientesRechazados <- dataTest %>%
  filter(loan_status == " Rejected" & education == " Not Graduate")

clientesModificados <- clientesRechazados
clientesModificados$education <- " Graduate"

prediccionesModificadas <- predict(mejorArbol, clientesModificados, type = "class")

seranAprobados <- which(prediccionesModificadas == " Approved")
impactoClientes <- clientesRechazados[seranAprobados, ]

cat("\nClientes que serían aprobados cambiando su educación a 'Graduate':\n")
print(paste("Total de clientes impactados:", nrow(impactoClientes)))
if(nrow(impactoClientes) > 0) {
  print(head(impactoClientes, 10))  
}

clientesRechazados <- dataTest %>%
  filter(loan_status == " Rejected")

encontrarMinimosNecesarios <- function(client, model, step = 1000000, max_increase = 10000000) {
  clienteTest <- client
  ingresoOrig <- client$income_annum

  for(income_increase in seq(0, max_increase, by = step)) {
    clienteTest$income_annum <- ingresoOrig + income_increase
    prediction <- predict(model, clienteTest, type = "class")

    if(prediction == " Approved") {
      return(clienteTest$income_annum)
    }
  }

  return(NA)
}

set.seed(123)  
tamanioMuestra <- min(20, nrow(clientesRechazados))
clientesMuestra <- clientesRechazados[sample(nrow(clientesRechazados), tamanioMuestra), ]

resultadoIngresos <- data.frame(
  client_id = clientesMuestra$loan_id,
  current_income = clientesMuestra$income_annum,
  min_income_needed = NA
)

for(i in 1:nrow(clientesMuestra)) {
  resultadoIngresos$min_income_needed[i] <- encontrarMinimosNecesarios(clientesMuestra[i,], mejorArbol)
}

resultadoIngresos$income_increase <- resultadoIngresos$min_income_needed - resultadoIngresos$current_income
resultadoIngresos$percentage_increase <- (resultadoIngresos$income_increase / resultadoIngresos$current_income) * 100

cat("\nIngresos mínimos necesarios para clientes con préstamo rechazado:\n")
print(resultadoIngresos)

clientesAprobados <- dataTest %>%
  filter(loan_status == " Approved")

encontrarMaxPrestamo <- function(client, model, step = 1000000, max_increase = 20000000) {
  clienteTest <- client
  prestamoOrig <- client$loan_amount

  ultimoAprobado <- prestamoOrig

  for(loan_increase in seq(step, max_increase, by = step)) {
    clienteTest$loan_amount <- prestamoOrig + loan_increase
    prediction <- predict(model, clienteTest, type = "class")

    if(prediction == " Rejected") {
      return(ultimoAprobado) 
    } else {
      ultimoAprobado <- clienteTest$loan_amount
    }
  }

  return(prestamoOrig + max_increase)
}

set.seed(456) 
tamanioMuestra <- min(20, nrow(clientesAprobados))
muestraAprobado <- clientesAprobados[sample(nrow(clientesAprobados), tamanioMuestra), ]

resultadosPrestamo <- data.frame(
  client_id = muestraAprobado$loan_id,
  current_loan = muestraAprobado$loan_amount,
  max_loan_possible = NA
)

for(i in 1:nrow(muestraAprobado)) {
  resultadosPrestamo$max_loan_possible[i] <- encontrarMaxPrestamo(muestraAprobado[i,], mejorArbol)
}

resultadosPrestamo$loan_increase <- resultadosPrestamo$max_loan_possible - resultadosPrestamo$current_loan
resultadosPrestamo$percentage_increase <- (resultadosPrestamo$loan_increase / resultadosPrestamo$current_loan) * 100

cat("\nMonto máximo de préstamo posible para clientes aprobados:\n")
print(resultadosPrestamo)
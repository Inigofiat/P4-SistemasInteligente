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

# Revisar estructura de los datos
str(data)
summary(data)

# Verificar valores faltantes
valoresFaltantes <- colSums(is.na(data))
print("Valores faltantes por columna:")
print(valoresFaltantes)

# ---------------------------------------------------------------
# 2. Análisis de Rangos para Préstamos Bancarios
# ---------------------------------------------------------------

# Verificar la idoneidad del uso de todos los atributos del dataset
# Identificar variables numéricas del dataset
varNum <- sapply(data, is.numeric)
columnasNum <- names(data)[varNum]
print("Variables numéricas en el dataset:")
print(columnasNum)

# Estadísticas descriptivas de variables numéricas
summary(data[, columnasNum])

# Función para visualizar la relación entre variables numéricas y loan_status
relacionNum <- function(data, var_name) {
  p <- ggplot(data, aes(x = .data[[var_name]], y = loan_status)) + 
    geom_point(alpha = 0.5) +
    labs(title = paste("Relación entre", var_name, "y aprobación de préstamo"),
         x = var_name, y = "Estado del préstamo") +
    theme_minimal()
  
  # Añadir anotación sobre rangos - especificando el valor de y correctamente
  if(var_name == "income_annum") {
    # Usar un valor de y que exista en los niveles de loan_status
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

# Visualizar relaciones para variables numéricas importantes
varImp <- c("income_annum", "loan_amount", "loan_term", "cibil_score")
plots <- list()

for(var in varImp) {
  dev.new()
  plots[[var]] <- relacionNum(data, var)
}

# Crear rangos para income_annum (ingresos anuales)
cat("\nCreando rangos para income_annum con umbrales constantes:\n")
data$income_range <- ifelse(data$income_annum <= 2500000, "Bajo", 
                            ifelse(data$income_annum <= 5000000, "Medio", "Alto"))

# Crear rangos para cibil_score (puntuación crediticia)
cat("\nCreando rangos para cibil_score con umbrales constantes:\n")
data$cibil_range <- ifelse(data$cibil_score <= 600, "Riesgo Alto",
                           ifelse(data$cibil_score <= 750, "Riesgo Medio", "Riesgo Bajo"))

# Verificar la distribución de los nuevos rangos
print("Distribución de rangos de ingresos:")
print(table(data$income_range))

print("Distribución de rangos de puntuación crediticia:")
print(table(data$cibil_range))

# Analizar la relación entre los rangos creados y la aprobación de préstamos
print("Relación entre rangos de ingresos y aprobación de préstamos:")
print(table(data$income_range, data$loan_status))

print("Relación entre rangos de puntuación crediticia y aprobación de préstamos:")
print(table(data$cibil_range, data$loan_status))

# Crear rangos uniformes para loan_amount (monto del préstamo)
cat("\nCreando rangos uniformes para loan_amount:\n")
data$loan_amount_range <- cut(data$loan_amount, breaks = 4)

# Crear rangos uniformes para loan_term (plazo del préstamo)
cat("\nCreando rangos uniformes para loan_term:\n")
data$loan_term_range <- cut(data$loan_term, breaks = 3)

# Verificar la distribución de los nuevos rangos uniformes
print("Distribución de rangos de monto de préstamo:")
print(table(data$loan_amount_range))

print("Distribución de rangos de plazo de préstamo:")
print(table(data$loan_term_range))

# Analizar la relación entre los rangos uniformes y la aprobación de préstamos
print("Relación entre rangos de monto de préstamo y aprobación:")
print(table(data$loan_amount_range, data$loan_status))

print("Relación entre rangos de plazo de préstamo y aprobación:")
print(table(data$loan_term_range, data$loan_status))

# Visualizar la relación entre los rangos de ingresos y la aprobación de préstamos
ingresosPrestamos <- as.data.frame(prop.table(table(data$income_range, data$loan_status), margin = 1) * 100)
colnames(ingresosPrestamos) <- c("Income_Range", "Loan_Status", "Percentage")

# Gráfico para rangos de ingresos
dev.new()
ggplot(ingresosPrestamos, aes(x = Income_Range, y = Percentage, fill = Loan_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Porcentaje de préstamos aprobados por rango de ingresos",
       x = "Rango de ingresos", y = "Porcentaje") +
  theme_minimal()

# Visualizar la relación entre los rangos de puntuación crediticia y la aprobación de préstamos
pCredPrestamos <- as.data.frame(prop.table(table(data$cibil_range, data$loan_status), margin = 1) * 100)
colnames(pCredPrestamos) <- c("Cibil_Range", "Loan_Status", "Percentage")

# Gráfico para rangos de puntuación crediticia
dev.new()
ggplot(pCredPrestamos, aes(x = Cibil_Range, y = Percentage, fill = Loan_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Porcentaje de préstamos aprobados por rango de puntuación crediticia",
       x = "Rango de puntuación crediticia", y = "Porcentaje") +
  theme_minimal()

# ---------------------------------------------------------------
# 3. Preprocesamiento de datos
# ---------------------------------------------------------------

# Reemplazar valores faltantes con el valor más común de cada columna
for(col in colnames(data)) {
  if(sum(is.na(data[[col]])) > 0) {
    if(is.numeric(data[[col]])) {
      # Para columnas numéricas, usar la mediana
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    } else {
      # Para columnas categóricas, usar la moda
      mode_val <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
      data[[col]][is.na(data[[col]])] <- mode_val
    }
  }
}

# Convertir variables categóricas a factores si no lo son
varCategoricas <- c("education", "self_employed", "loan_status")
for(var in varCategoricas) {
  if(var %in% colnames(data)) {
    data[[var]] <- as.factor(data[[var]])
  }
}

# ---------------------------------------------------------------
# 4. Análisis exploratorio de datos
# ---------------------------------------------------------------

# Distribución de la variable objetivo
print("Distribución de aprobación de préstamos:")
table(data$loan_status)

# Relación entre educación y aprobación de préstamos
print("Relación entre educación y aprobación de préstamos:")
table(data$education, data$loan_status)

# Relación entre autoempleo y aprobación de préstamos
print("Relación entre autoempleo y aprobación de préstamos:")
table(data$self_employed, data$loan_status)

# Visualización de variables numéricas relevantes
varNum <- c("income_annum", "loan_amount", "loan_term", "cibil_score")

# Histogramas para variables numéricas
dev.new()
par(mfrow=c(2,2))
for(var in varNum) {
  if(var %in% colnames(data)) {
    hist(data[[var]], main=paste("Histograma de", var), xlab=var, col="lightblue")
  }
}
par(mfrow=c(1,1))

# ---------------------------------------------------------------
# 5. Evaluar si los rangos mejoran el rendimiento del modelo
# ---------------------------------------------------------------

# Función para entrenar y evaluar un modelo con variables originales
varOrig <- function() {
  set.seed(123)
  indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  dataPrueba <- data[indexPrueba, ]
  dataTest <- data[-indexPrueba, ]
  
  # Variables originales
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

# Función para entrenar y evaluar un modelo con variables por rangos
pruebaRangos <- function() {
  set.seed(123)
  indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  dataPrueba <- data[indexPrueba, ]
  dataTest <- data[-indexPrueba, ]
  
  # Variables con rangos
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

# Evaluar modelos
cat("\nEvaluando rendimiento de modelos con variables originales vs rangos...\n")
resultadosOrig <- varOrig()
resultadosRangos <- pruebaRangos()

# Comparar resultados
cat("\nResultados de evaluación:\n")
cat("Precisión modelo con variables originales:", round(resultadosOrig$accuracy, 4), "\n")
cat("Precisión modelo con variables por rangos:", round(resultadosRangos$accuracy, 4), "\n")

# ---------------------------------------------------------------
# 6. División de datos en entrenamiento y prueba
# ---------------------------------------------------------------

# Establecer semilla para reproducibilidad
set.seed(123)

# Dividir los datos: 75% entrenamiento, 25% prueba
indexPrueba <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
dataTrain <- data[indexPrueba, ]
dataTest <- data[-indexPrueba, ]

# ---------------------------------------------------------------
# 7. Modelado: Creación de árboles de decisión
# ---------------------------------------------------------------

# Modificación principal: Función para entrenar un árbol de decisión con variabilidad
arbolPrueba <- function(dataTrain, dataTest, max_depth = 5, min_split = 20, cp = 0.01) {
  # Entrenar modelo con parámetros variables
  modeloArbol <- rpart(loan_status ~ ., 
                       data = dataTrain, 
                       method = "class",
                       control = rpart.control(maxdepth = max_depth, 
                                               minsplit = min_split,
                                               cp = cp))
  
  # Hacer predicciones
  predicciones <- predict(modeloArbol, dataTest, type = "class")
  
  # Evaluar modelo
  matrixConf <- confusionMatrix(predicciones, dataTest$loan_status)
  
  # Calcular métricas específicas
  accuracy <- matrixConf$overall["Accuracy"]
  
  # Calcular precisión para cada clase
  # Tratar los casos donde algunas métricas podrían no estar disponibles
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
  
  # Retornar resultados
  return(list(
    model = modeloArbol,
    predicciones = predicciones,
    matrixConf = matrixConf,
    accuracy = accuracy,
    precisionAprobada = precisionAprobada,
    precisionDenegada = precisionDenegada
  ))
}

# Crear y evaluar 10 árboles con diferentes configuraciones para garantizar variabilidad
resultados <- list()

# Configuraciones diferentes para cada árbol
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

# También usaremos diferentes subconjuntos de datos para cada árbol
set.seed(123)
for(i in 1:10) {
  cat("Entrenando árbol", i, "...\n")
  
  # Usar diferentes semillas para cada árbol
  set.seed(i * 100)
  
  # Seleccionar un subconjunto aleatorio de datos (80% de los datos de entrenamiento)
  muestra_indices <- sample(1:nrow(dataTrain), size = floor(0.8 * nrow(dataTrain)))
  muestra_train <- dataTrain[muestra_indices, ]
  
  # Entrenar con diferentes configuraciones
  resultados[[i]] <- arbolPrueba(
    muestra_train, 
    dataTest,
    max_depth = configuraciones[[i]]$max_depth,
    min_split = configuraciones[[i]]$min_split,
    cp = configuraciones[[i]]$cp
  )
}

# ---------------------------------------------------------------
# 8. Encontrar el mejor árbol de decisión
# ---------------------------------------------------------------

# Extraer precisiones
accuracies <- sapply(resultados, function(r) r$accuracy)
print("Precisiones de los 10 árboles:")
print(round(accuracies, 4))

# Encontrar el índice del mejor modelo
mejorModeloIndex <- which.max(accuracies)
mejorModelo <- resultados[[mejorModeloIndex]]

# Visualizar el mejor árbol
dev.new()
print(paste("El mejor árbol es el #", mejorModeloIndex, "con precisión:", round(mejorModelo$accuracy, 4)))
rpart.plot(mejorModelo$model, extra = 104, box.palette = "RdBu", shadow.col = "gray")

# Generar reglas del mejor árbol
reglas <- rpart.rules(mejorModelo$model, roundint = FALSE)
print("Reglas que definen el comportamiento del árbol:")
print(reglas)

# ---------------------------------------------------------------
# 9. Evaluar importancia de atributos
# ---------------------------------------------------------------

# Calcular importancia de variables usando Random Forest como complemento
set.seed(123)
modeloRf <- randomForest(loan_status ~ ., data = dataTrain)
varImportante <- importance(modeloRf)
varImportanteDf <- data.frame(
  Variable = row.names(varImportante),
  Importance = varImportante[,1]
)
varImportanteDf <- varImportanteDf[order(varImportanteDf$Importance, decreasing = TRUE),]

# Mostrar los 5 atributos más relevantes
print("Los 5 atributos más relevantes:")
print(head(varImportanteDf, 5))

# Visualizar importancia de variables
dev.new()
barplot(varImportanteDf$Importance[1:5], 
        names.arg = varImportanteDf$Variable[1:5],
        col = "steelblue",
        main = "Importancia de Variables",
        ylab = "Importancia",
        las = 2)

# ---------------------------------------------------------------
# 10. Evaluar precisión por grupos específicos
# ---------------------------------------------------------------

# Evaluar precisión considerando solo la educación
educationAccuracy <- mean(mejorModelo$predicciones[dataTest$education == " Graduate"] == 
                            dataTest$loan_status[dataTest$education == " Graduate"])
print(paste("Precisión considerando solo educación (Graduate):", round(educationAccuracy, 4)))

# Evaluar precisión considerando solo si es trabajador autónomo
accuracyTrabajadorAuto <- mean(mejorModelo$predicciones[dataTest$self_employed == " Yes"] == 
                                 dataTest$loan_status[dataTest$self_employed == " Yes"])
print(paste("Precisión considerando solo trabajadores autónomos:", round(accuracyTrabajadorAuto, 4)))

# ---------------------------------------------------------------
# 11. Tabla de resultados
# ---------------------------------------------------------------

# Crear tabla con los resultados de los 10 árboles
resultadosTabla <- data.frame(
  Tree = 1:10,
  Accuracy = sapply(resultados, function(r) r$accuracy),
  PrecisionApproved = sapply(resultados, function(r) r$precisionAprobada),
  PrecisionRejected = sapply(resultados, function(r) r$precisionDenegada)
)

# Calcular medias
resultadosTabla <- rbind(resultadosTabla, 
                         data.frame(Tree = "Media", 
                                    Accuracy = mean(resultadosTabla$Accuracy),
                                    PrecisionApproved = mean(resultadosTabla$PrecisionApproved, na.rm = TRUE),
                                    PrecisionRejected = mean(resultadosTabla$PrecisionRejected, na.rm = TRUE)))

# Mostrar tabla de resultados
print("Tabla de resultados de los 10 árboles:")
print(resultadosTabla, digits = 4)

# Abrir nuevo dispositivo gráfico para el gráfico
dev.new()

# Convertir a formato largo para graficar (solo las primeras 10 filas, sin la media)
resultadosLong <- melt(resultadosTabla[1:10,], id.vars = "Tree", 
                       variable.name = "Metric", value.name = "Value")

# Crear gráfico
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

# ---------------------------------------------------------------
# 12. Conclusiones sobre el análisis de rangos
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------
# 13. Conclusiones generales del modelado
# ---------------------------------------------------------------
cat("\nCONCLUSIONES GENERALES DEL MODELADO:\n")
cat("- Se han generado 10 árboles de decisión con diferentes configuraciones para predecir la aprobación de préstamos.\n")
cat(paste("- La precisión media de los modelos es del:", round(mean(accuracies) * 100, 2), "%\n"))
cat(paste("- El mejor modelo (árbol #", mejorModeloIndex, ") tiene una precisión del:", 
          round(mejorModelo$accuracy * 100, 2), "%\n"))
cat("- Las variables más relevantes para la predicción son:\n")
for(i in 1:5) {
  cat(paste("  ", i, ". ", varImportanteDf$Variable[i], "\n"))
}

# ---------------------------------------------------------------
# 14. Respuestas a las preguntas solicitadas
# ---------------------------------------------------------------

# Usaremos el mejor modelo identificado previamente
mejorArbol <- mejorModelo$model

# Pregunta 1: Generar una representación gráfica sobre porcentajes de valores actuales y 
# predichos por el modelo por número de dependientes

# Extraer datos de test para análisis
dataTestPredic <- dataTest
dataTestPredic$predicted_status <- mejorModelo$predicciones

# Crear un dataframe con los resultados agrupados por número de dependientes
analisisDependiente <- dataTest %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    actual_approved = sum(loan_status == " Approved"),
    actual_approved_pct = actual_approved / total * 100
  )

# Añadir predicciones por grupo de dependientes
prediccionesDependientes <- dataTestPredic %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    predicted_approved = sum(predicted_status == " Approved"),
    predicted_approved_pct = predicted_approved / total * 100
  )

# Combinar ambos dataframes
resultadosDependientes <- merge(analisisDependiente, prediccionesDependientes, by="no_of_dependents")
resultadosDependientes <- resultadosDependientes %>%
  select(no_of_dependents, actual_approved_pct, predicted_approved_pct)

# Convertir a formato largo para graficar
longDependientes <- melt(resultadosDependientes, 
                         id.vars = "no_of_dependents",
                         variable.name = "status_type", 
                         value.name = "percentage")

# Crear gráfico
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

# Pregunta 2: Identificar clientes con préstamo rechazado tal que, si cambiamos 
# su educación de 'Not Graduate' a 'Graduate', les hubiesen concedido el crédito

# Filtrar clientes rechazados con educación 'Not Graduate'
clientesRechazados <- dataTest %>%
  filter(loan_status == " Rejected" & education == " Not Graduate")

# Crear un dataframe con estos clientes pero cambiando su educación a 'Graduate'
clientesModificados <- clientesRechazados
clientesModificados$education <- " Graduate"

# Predecir con el modelo para estos clientes modificados
prediccionesModificadas <- predict(mejorArbol, clientesModificados, type = "class")

# Identificar aquellos que serían aprobados con el cambio
seranAprobados <- which(prediccionesModificadas == " Approved")
impactoClientes <- clientesRechazados[seranAprobados, ]

# Mostrar resultados
cat("\nClientes que serían aprobados cambiando su educación a 'Graduate':\n")
print(paste("Total de clientes impactados:", nrow(impactoClientes)))
if(nrow(impactoClientes) > 0) {
  print(head(impactoClientes, 10))  # Mostrar los primeros 10 como ejemplo
}

# Pregunta 3: Para cada cliente con crédito rechazado, identificar los ingresos
# mínimos con los que le habrían aprobado el préstamo

# Filtrar clientes rechazados
clientesRechazados <- dataTest %>%
  filter(loan_status == " Rejected")

# Función para encontrar ingresos mínimos necesarios para aprobación
encontrarMinimosNecesarios <- function(client, model, step = 1000000, max_increase = 10000000) {
# Copiar el cliente original
  clienteTest <- client
  ingresoOrig <- client$income_annum

# Incrementar ingresos hasta que sea aprobado o se alcance el máximo
  for(income_increase in seq(0, max_increase, by = step)) {
    clienteTest$income_annum <- ingresoOrig + income_increase
    prediction <- predict(model, clienteTest, type = "class")

    if(prediction == " Approved") {
      return(clienteTest$income_annum)
    }
  }

# Si no se encuentra una solución, devolver NA
  return(NA)
}

# # Calcular ingresos mínimos para cada cliente rechazado (limitado a 20 para eficiencia)
set.seed(123)  # Para reproducibilidad
tamanioMuestra <- min(20, nrow(clientesRechazados))
clientesMuestra <- clientesRechazados[sample(nrow(clientesRechazados), tamanioMuestra), ]

resultadoIngresos <- data.frame(
  client_id = clientesMuestra$loan_id,
  current_income = clientesMuestra$income_annum,
  min_income_needed = NA
)

# # Calcular para cada cliente en la muestra
for(i in 1:nrow(clientesMuestra)) {
  resultadoIngresos$min_income_needed[i] <- encontrarMinimosNecesarios(clientesMuestra[i,], mejorArbol)
}

# Calcular incremento necesario
resultadoIngresos$income_increase <- resultadoIngresos$min_income_needed - resultadoIngresos$current_income
resultadoIngresos$percentage_increase <- (resultadoIngresos$income_increase / resultadoIngresos$current_income) * 100

# Mostrar resultados
cat("\nIngresos mínimos necesarios para clientes con préstamo rechazado:\n")
print(resultadoIngresos)

# Pregunta 4: Para cada cliente con crédito aprobado, identificar la máxima cantidad
# de préstamo que podría haber pedido, y aún haber sido aprobado

#Filtrar clientes aprobados
clientesAprobados <- dataTest %>%
  filter(loan_status == " Approved")

# Función para encontrar el préstamo máximo que sería aprobado
encontrarMaxPrestamo <- function(client, model, step = 1000000, max_increase = 20000000) {
  # Copiar el cliente original
  clienteTest <- client
  prestamoOrig <- client$loan_amount

  # Incrementar monto del préstamo hasta que sea rechazado o se alcance el máximo
  ultimoAprobado <- prestamoOrig

  for(loan_increase in seq(step, max_increase, by = step)) {
    clienteTest$loan_amount <- prestamoOrig + loan_increase
    prediction <- predict(model, clienteTest, type = "class")

    if(prediction == " Rejected") {
      return(ultimoAprobado)  # Devolver el último monto aprobado
    } else {
      ultimoAprobado <- clienteTest$loan_amount
    }
  }

  # Si todas las pruebas son aprobadas, devolver el máximo
  return(prestamoOrig + max_increase)
}

# Calcular préstamo máximo para una muestra de clientes aprobados
set.seed(456)  # Para reproducibilidad
tamanioMuestra <- min(20, nrow(clientesAprobados))
muestraAprobado <- clientesAprobados[sample(nrow(clientesAprobados), tamanioMuestra), ]

resultadosPrestamo <- data.frame(
  client_id = muestraAprobado$loan_id,
  current_loan = muestraAprobado$loan_amount,
  max_loan_possible = NA
)

# Calcular para cada cliente en la muestra
for(i in 1:nrow(muestraAprobado)) {
  resultadosPrestamo$max_loan_possible[i] <- encontrarMaxPrestamo(muestraAprobado[i,], mejorArbol)
}

# Calcular incremento posible
resultadosPrestamo$loan_increase <- resultadosPrestamo$max_loan_possible - resultadosPrestamo$current_loan
resultadosPrestamo$percentage_increase <- (resultadosPrestamo$loan_increase / resultadosPrestamo$current_loan) * 100

# Mostrar resultados
cat("\nMonto máximo de préstamo posible para clientes aprobados:\n")
print(resultadosPrestamo)
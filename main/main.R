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
numeric_columns <- names(data)[varNum]
print("Variables numéricas en el dataset:")
print(numeric_columns)

# Estadísticas descriptivas de variables numéricas
summary(data[, numeric_columns])

# Función para visualizar la relación entre variables numéricas y loan_status
visualize_numeric_relationship <- function(data, var_name) {
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
key_numeric_vars <- c("income_annum", "loan_amount", "loan_term", "cibil_score")
plots <- list()

for(var in key_numeric_vars) {
  dev.new()
  plots[[var]] <- visualize_numeric_relationship(data, var)
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
income_approval <- as.data.frame(prop.table(table(data$income_range, data$loan_status), margin = 1) * 100)
colnames(income_approval) <- c("Income_Range", "Loan_Status", "Percentage")

# Gráfico para rangos de ingresos
dev.new()
ggplot(income_approval, aes(x = Income_Range, y = Percentage, fill = Loan_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Porcentaje de préstamos aprobados por rango de ingresos",
       x = "Rango de ingresos", y = "Porcentaje") +
  theme_minimal()

# Visualizar la relación entre los rangos de puntuación crediticia y la aprobación de préstamos
cibil_approval <- as.data.frame(prop.table(table(data$cibil_range, data$loan_status), margin = 1) * 100)
colnames(cibil_approval) <- c("Cibil_Range", "Loan_Status", "Percentage")

# Gráfico para rangos de puntuación crediticia
dev.new()
ggplot(cibil_approval, aes(x = Cibil_Range, y = Percentage, fill = Loan_Status)) +
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
categorical_vars <- c("education", "self_employed", "loan_status")
for(var in categorical_vars) {
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
train_evaluate_original <- function() {
  set.seed(123)
  train_index <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Variables originales
  formula_orig <- loan_status ~ income_annum + loan_amount + loan_term + cibil_score + 
    education + self_employed + no_of_dependents
  
  model_orig <- rpart(formula_orig, data = train_data, method = "class", 
                      control = rpart.control(maxdepth = 5))
  
  pred_orig <- predict(model_orig, test_data, type = "class")
  conf_matrix_orig <- confusionMatrix(pred_orig, test_data$loan_status)
  
  return(list(
    accuracy = conf_matrix_orig$overall["Accuracy"],
    model = model_orig
  ))
}

# Función para entrenar y evaluar un modelo con variables por rangos
train_evaluate_ranges <- function() {
  set.seed(123)
  train_index <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Variables con rangos
  formula_ranges <- loan_status ~ income_range + cibil_range + loan_amount_range + 
    loan_term_range + education + self_employed + no_of_dependents
  
  model_ranges <- rpart(formula_ranges, data = train_data, method = "class", 
                        control = rpart.control(maxdepth = 5))
  
  pred_ranges <- predict(model_ranges, test_data, type = "class")
  conf_matrix_ranges <- confusionMatrix(pred_ranges, test_data$loan_status)
  
  return(list(
    accuracy = conf_matrix_ranges$overall["Accuracy"],
    model = model_ranges
  ))
}

# Evaluar modelos
cat("\nEvaluando rendimiento de modelos con variables originales vs rangos...\n")
original_results <- train_evaluate_original()
ranges_results <- train_evaluate_ranges()

# Comparar resultados
cat("\nResultados de evaluación:\n")
cat("Precisión modelo con variables originales:", round(original_results$accuracy, 4), "\n")
cat("Precisión modelo con variables por rangos:", round(ranges_results$accuracy, 4), "\n")

# ---------------------------------------------------------------
# 6. División de datos en entrenamiento y prueba
# ---------------------------------------------------------------

# Establecer semilla para reproducibilidad
set.seed(123)

# Dividir los datos: 75% entrenamiento, 25% prueba
train_index <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# ---------------------------------------------------------------
# 7. Modelado: Creación de árboles de decisión
# ---------------------------------------------------------------

# Función para entrenar un árbol de decisión, hacer predicciones y evaluar
train_and_evaluate_tree <- function(train_data, test_data, max_depth = 5) {
  # Entrenar modelo
  tree_model <- rpart(loan_status ~ ., 
                      data = train_data, 
                      method = "class",
                      control = rpart.control(maxdepth = max_depth))
  
  # Hacer predicciones
  predictions <- predict(tree_model, test_data, type = "class")
  
  # Evaluar modelo
  confusion_matrix <- confusionMatrix(predictions, test_data$loan_status)
  
  # Calcular métricas específicas
  accuracy <- confusion_matrix$overall["Accuracy"]
  precision_approved <- confusion_matrix$byClass["Pos Pred Value"]
  precision_rejected <- confusion_matrix$byClass["Neg Pred Value"]
  
  # Retornar resultados
  return(list(
    model = tree_model,
    predictions = predictions,
    confusion_matrix = confusion_matrix,
    accuracy = accuracy,
    precision_approved = precision_approved,
    precision_rejected = precision_rejected
  ))
}

# Crear y evaluar 10 árboles
results <- list()
for(i in 1:10) {
  set.seed(i * 100)  # Usar semillas diferentes para cada iteración
  cat("Entrenando árbol", i, "...\n")
  results[[i]] <- train_and_evaluate_tree(train_data, test_data)
}

# ---------------------------------------------------------------
# 8. Encontrar el mejor árbol de decisión
# ---------------------------------------------------------------

# Extraer precisiones
accuracies <- sapply(results, function(r) r$accuracy)

# Encontrar el índice del mejor modelo
best_model_index <- which.max(accuracies)
best_model <- results[[best_model_index]]

# Visualizar el mejor árbol
dev.new()
print(paste("El mejor árbol es el #", best_model_index, "con precisión:", round(best_model$accuracy, 4)))
rpart.plot(best_model$model, extra = 104, box.palette = "RdBu", shadow.col = "gray")

# Generar reglas del mejor árbol
rules <- rpart.rules(best_model$model, roundint = FALSE)
print("Reglas que definen el comportamiento del árbol:")
print(rules)

# ---------------------------------------------------------------
# 9. Evaluar importancia de atributos
# ---------------------------------------------------------------

# Calcular importancia de variables usando Random Forest como complemento
set.seed(123)
rf_model <- randomForest(loan_status ~ ., data = train_data)
var_importance <- importance(rf_model)
var_importance_df <- data.frame(
  Variable = row.names(var_importance),
  Importance = var_importance[,1]
)
var_importance_df <- var_importance_df[order(var_importance_df$Importance, decreasing = TRUE),]

# Mostrar los 5 atributos más relevantes
print("Los 5 atributos más relevantes:")
print(head(var_importance_df, 5))

# Visualizar importancia de variables
dev.new()
barplot(var_importance_df$Importance[1:5], 
        names.arg = var_importance_df$Variable[1:5],
        col = "steelblue",
        main = "Importancia de Variables",
        ylab = "Importancia",
        las = 2)

# ---------------------------------------------------------------
# 10. Evaluar precisión por grupos específicos
# ---------------------------------------------------------------

# Evaluar precisión considerando solo la educación
education_accuracy <- mean(best_model$predictions[test_data$education == " Graduate"] == 
                             test_data$loan_status[test_data$education == " Graduate"])
print(paste("Precisión considerando solo educación (Graduate):", round(education_accuracy, 4)))

# Evaluar precisión considerando solo si es trabajador autónomo
self_employed_accuracy <- mean(best_model$predictions[test_data$self_employed == " Yes"] == 
                                 test_data$loan_status[test_data$self_employed == " Yes"])
print(paste("Precisión considerando solo trabajadores autónomos:", round(self_employed_accuracy, 4)))

# ---------------------------------------------------------------
# 11. Tabla de resultados
# ---------------------------------------------------------------

# Crear tabla con los resultados de los 10 árboles
results_table <- data.frame(
  Tree = 1:10,
  Accuracy = sapply(results, function(r) r$accuracy),
  PrecisionApproved = sapply(results, function(r) r$precision_approved),
  PrecisionRejected = sapply(results, function(r) r$precision_rejected)
)

# Calcular medias
results_table <- rbind(results_table, 
                       data.frame(Tree = "Media", 
                                  Accuracy = mean(results_table$Accuracy),
                                  PrecisionApproved = mean(results_table$PrecisionApproved),
                                  PrecisionRejected = mean(results_table$PrecisionRejected)))

# Mostrar tabla de resultados
print("Tabla de resultados de los 10 árboles:")
print(results_table, digits = 4)

# Abrir nuevo dispositivo gráfico para el gráfico
dev.new()

# Convertir a formato largo para graficar (solo las primeras 10 filas, sin la media)
results_long <- melt(results_table[1:10,], id.vars = "Tree", 
                     variable.name = "Metric", value.name = "Value")

# Crear gráfico
ggplot(results_long, aes(x = factor(Tree), y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(data = results_table[11,], 
             aes(yintercept = Accuracy, color = "Media Accuracy"), 
             linetype = "dashed") +
  geom_hline(data = results_table[11,], 
             aes(yintercept = PrecisionApproved, color = "Media Precision Approved"), 
             linetype = "dashed") +
  geom_hline(data = results_table[11,], 
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

if(ranges_results$accuracy > original_results$accuracy) {
  cat("4. El modelo con variables categorizadas en rangos mejora la precisión en ", 
      round((ranges_results$accuracy - original_results$accuracy) * 100, 2), 
      "% respecto al modelo con variables originales.\n")
  cat("5. La transformación de variables numéricas a rangos parece beneficiar el rendimiento del modelo.\n")
} else {
  cat("4. El modelo con variables originales tiene mejor precisión que el modelo con rangos (diferencia de ", 
      round((original_results$accuracy - ranges_results$accuracy) * 100, 2), "%).\n")
  cat("5. La transformación de variables numéricas a rangos no parece beneficiar el rendimiento del modelo en este caso.\n")
}

cat("6. Los rangos creados ofrecen mayor interpretabilidad a costa de posible pérdida de información detallada.\n")

# ---------------------------------------------------------------
# 13. Conclusiones generales del modelado
# ---------------------------------------------------------------
cat("\nCONCLUSIONES GENERALES DEL MODELADO:\n")
cat("- Se han generado 10 árboles de decisión para predecir la aprobación de préstamos.\n")
cat(paste("- La precisión media de los modelos es del:", round(mean(accuracies) * 100, 2), "%\n"))
cat(paste("- El mejor modelo (árbol #", best_model_index, ") tiene una precisión del:", 
          round(best_model$accuracy * 100, 2), "%\n"))
cat("- Las variables más relevantes para la predicción son:\n")
for(i in 1:5) {
  cat(paste("  ", i, ". ", var_importance_df$Variable[i], "\n"))
}

# ---------------------------------------------------------------
# 14. Respuestas a las preguntas solicitadas
# ---------------------------------------------------------------

# Usaremos el mejor modelo identificado previamente
best_tree <- best_model$model

# Pregunta 1: Generar una representación gráfica sobre porcentajes de valores actuales y 
# predichos por el modelo por número de dependientes

# Extraer datos de test para análisis
test_data_with_predictions <- test_data
test_data_with_predictions$predicted_status <- best_model$predictions

# Crear un dataframe con los resultados agrupados por número de dependientes
dependents_analysis <- test_data %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    actual_approved = sum(loan_status == " Approved"),
    actual_approved_pct = actual_approved / total * 100
  )

# Añadir predicciones por grupo de dependientes
dependents_predictions <- test_data_with_predictions %>%
  group_by(no_of_dependents) %>%
  summarize(
    total = n(),
    predicted_approved = sum(predicted_status == " Approved"),
    predicted_approved_pct = predicted_approved / total * 100
  )

# Combinar ambos dataframes
dependents_results <- merge(dependents_analysis, dependents_predictions, by="no_of_dependents")
dependents_results <- dependents_results %>%
  select(no_of_dependents, actual_approved_pct, predicted_approved_pct)

# Convertir a formato largo para graficar
dependents_long <- melt(dependents_results, 
                        id.vars = "no_of_dependents",
                        variable.name = "status_type", 
                        value.name = "percentage")

# Crear gráfico
dev.new()
ggplot(dependents_long, aes(x=factor(no_of_dependents), y=percentage, fill=status_type)) +
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
rejected_clients <- test_data %>%
  filter(loan_status == " Rejected" & education == " Not Graduate")

# Crear un dataframe con estos clientes pero cambiando su educación a 'Graduate'
modified_clients <- rejected_clients
modified_clients$education <- " Graduate"

# Predecir con el modelo para estos clientes modificados
modified_predictions <- predict(best_tree, modified_clients, type = "class")

# Identificar aquellos que serían aprobados con el cambio
would_be_approved <- which(modified_predictions == " Approved")
clients_education_impact <- rejected_clients[would_be_approved, ]

# Mostrar resultados
cat("\nClientes que serían aprobados cambiando su educación a 'Graduate':\n")
print(paste("Total de clientes impactados:", nrow(clients_education_impact)))
if(nrow(clients_education_impact) > 0) {
  print(head(clients_education_impact, 10))  # Mostrar los primeros 10 como ejemplo
}

# Pregunta 3: Para cada cliente con crédito rechazado, identificar los ingresos 
# mínimos con los que le habrían aprobado el préstamo

# Filtrar clientes rechazados
rejected_clients <- test_data %>%
  filter(loan_status == " Rejected")

# Función para encontrar ingresos mínimos necesarios para aprobación
find_min_income <- function(client, model, step = 1000000, max_increase = 10000000) {
  # Copiar el cliente original
  test_client <- client
  original_income <- client$income_annum
  
  # Incrementar ingresos hasta que sea aprobado o se alcance el máximo
  for(income_increase in seq(0, max_increase, by = step)) {
    test_client$income_annum <- original_income + income_increase
    prediction <- predict(model, test_client, type = "class")
    
    if(prediction == " Approved") {
      return(test_client$income_annum)
    }
  }
  
  # Si no se encuentra una solución, devolver NA
  return(NA)
}

# Calcular ingresos mínimos para cada cliente rechazado (limitado a 20 para eficiencia)
set.seed(123)  # Para reproducibilidad
sample_size <- min(20, nrow(rejected_clients))
sampled_clients <- rejected_clients[sample(nrow(rejected_clients), sample_size), ]

income_results <- data.frame(
  client_id = sampled_clients$loan_id,
  current_income = sampled_clients$income_annum,
  min_income_needed = NA
)

# Calcular para cada cliente en la muestra
for(i in 1:nrow(sampled_clients)) {
  income_results$min_income_needed[i] <- find_min_income(sampled_clients[i,], best_tree)
}

# Calcular incremento necesario
income_results$income_increase <- income_results$min_income_needed - income_results$current_income
income_results$percentage_increase <- (income_results$income_increase / income_results$current_income) * 100

# Mostrar resultados
cat("\nIngresos mínimos necesarios para clientes con préstamo rechazado:\n")
print(income_results)

# Pregunta 4: Para cada cliente con crédito aprobado, identificar la máxima cantidad
# de préstamo que podría haber pedido, y aún haber sido aprobado

# Filtrar clientes aprobados
approved_clients <- test_data %>%
  filter(loan_status == " Approved")

# Función para encontrar el préstamo máximo que sería aprobado
find_max_loan <- function(client, model, step = 1000000, max_increase = 20000000) {
  # Copiar el cliente original
  test_client <- client
  original_loan <- client$loan_amount
  
  # Incrementar monto del préstamo hasta que sea rechazado o se alcance el máximo
  last_approved <- original_loan
  
  for(loan_increase in seq(step, max_increase, by = step)) {
    test_client$loan_amount <- original_loan + loan_increase
    prediction <- predict(model, test_client, type = "class")
    
    if(prediction == " Rejected") {
      return(last_approved)  # Devolver el último monto aprobado
    } else {
      last_approved <- test_client$loan_amount
    }
  }
  
  # Si todas las pruebas son aprobadas, devolver el máximo
  return(original_loan + max_increase)
}

# Calcular préstamo máximo para una muestra de clientes aprobados
set.seed(456)  # Para reproducibilidad
sample_size <- min(20, nrow(approved_clients))
sampled_approved <- approved_clients[sample(nrow(approved_clients), sample_size), ]

loan_results <- data.frame(
  client_id = sampled_approved$loan_id,
  current_loan = sampled_approved$loan_amount,
  max_loan_possible = NA
)

# Calcular para cada cliente en la muestra
for(i in 1:nrow(sampled_approved)) {
  loan_results$max_loan_possible[i] <- find_max_loan(sampled_approved[i,], best_tree)
}

# Calcular incremento posible
loan_results$loan_increase <- loan_results$max_loan_possible - loan_results$current_loan
loan_results$percentage_increase <- (loan_results$loan_increase / loan_results$current_loan) * 100

# Mostrar resultados
cat("\nMonto máximo de préstamo posible para clientes aprobados:\n")
print(loan_results)
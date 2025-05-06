# ---------------------------------------------------------------
# Predicción de Aprobación de Préstamos Bancarios
# ---------------------------------------------------------------

# Cargar librerías necesarias
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)

# Limpiar el ambiente de trabajo
rm(list=ls())
cat("\014")
if(!is.null(dev.list())) dev.off()
graphics.off()

# Configurar directorio de trabajo (ajustar según necesidades)
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ---------------------------------------------------------------
# 1. Cargar y explorar los datos
# ---------------------------------------------------------------
data <- read.csv("../data/2025_loan_approval_dataset.csv")

# Revisar estructura de los datos
str(data)
summary(data)

# Verificar valores faltantes
missing_values <- colSums(is.na(data))
print("Valores faltantes por columna:")
print(missing_values)

# ---------------------------------------------------------------
# 2. Preprocesamiento de datos
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
# 3. Análisis exploratorio de datos
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
numeric_vars <- c("income_annum", "loan_amount", "loan_term", "cibil_score")

# Histogramas para variables numéricas
par(mfrow=c(2,2))
for(var in numeric_vars) {
  if(var %in% colnames(data)) {
    hist(data[[var]], main=paste("Histograma de", var), xlab=var, col="lightblue")
  }
}
par(mfrow=c(1,1))

# ---------------------------------------------------------------
# 4. División de datos en entrenamiento y prueba
# ---------------------------------------------------------------

# Establecer semilla para reproducibilidad
set.seed(123)

# Dividir los datos: 75% entrenamiento, 25% prueba
train_index <- createDataPartition(data$loan_status, p = 0.75, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# ---------------------------------------------------------------
# 5. Modelado: Creación de árboles de decisión
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
# 6. Encontrar el mejor árbol de decisión
# ---------------------------------------------------------------

# Extraer precisiones
accuracies <- sapply(results, function(r) r$accuracy)

# Encontrar el índice del mejor modelo
best_model_index <- which.max(accuracies)
best_model <- results[[best_model_index]]

# Visualizar el mejor árbol
print(paste("El mejor árbol es el #", best_model_index, "con precisión:", round(best_model$accuracy, 4)))
rpart.plot(best_model$model, extra = 104, box.palette = "RdBu", shadow.col = "gray")

# Generar reglas del mejor árbol
rules <- rpart.rules(best_model$model, roundint = FALSE)
print("Reglas que definen el comportamiento del árbol:")
print(rules)

# ---------------------------------------------------------------
# 7. Evaluar importancia de atributos
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
barplot(var_importance_df$Importance[1:5], 
        names.arg = var_importance_df$Variable[1:5],
        col = "steelblue",
        main = "Importancia de Variables",
        ylab = "Importancia",
        las = 2)

# ---------------------------------------------------------------
# 8. Evaluar precisión por grupos específicos
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
# 9. Tabla de resultados
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

# ---------------------------------------------------------------
# 10. Conclusiones
# ---------------------------------------------------------------
cat("\nCONCLUSIONES:\n")
cat("- Se han generado 10 árboles de decisión para predecir la aprobación de préstamos.\n")
cat(paste("- La precisión media de los modelos es del:", round(mean(accuracies) * 100, 2), "%\n"))
cat(paste("- El mejor modelo (árbol #", best_model_index, ") tiene una precisión del:", 
          round(best_model$accuracy * 100, 2), "%\n"))
cat("- Las variables más relevantes para la predicción son:\n")
for(i in 1:5) {
  cat(paste("  ", i, ". ", var_importance_df$Variable[i], "\n"))
}

# ---------------------------------------------------------------
# 11. Respuestas a las preguntas solicitadas
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
  group_by(no_of_dependents) %>%  # Corregido: era "dependents"
  summarize(
    total = n(),
    actual_approved = sum(loan_status == " Approved"),  # Nota el espacio antes de "Approved"
    actual_approved_pct = actual_approved / total * 100
  )

# Añadir predicciones por grupo de dependientes
dependents_predictions <- test_data_with_predictions %>%
  group_by(no_of_dependents) %>%  # Corregido: era "dependents"
  summarize(
    total = n(),
    predicted_approved = sum(predicted_status == " Approved"),  # Nota el espacio antes de "Approved"
    predicted_approved_pct = predicted_approved / total * 100
  )

# Combinar ambos dataframes
dependents_results <- merge(dependents_analysis, dependents_predictions, by="no_of_dependents")  # Corregido: era "dependents"
dependents_results <- dependents_results %>%
  select(no_of_dependents, actual_approved_pct, predicted_approved_pct)  # Corregido: era "dependents"

# Convertir a formato largo para graficar
library(reshape2)
dependents_long <- melt(dependents_results, 
                        id.vars = "no_of_dependents",  # Corregido: era "dependents"
                        variable.name = "status_type", 
                        value.name = "percentage")

# Crear gráfico
ggplot(dependents_long, aes(x=factor(no_of_dependents), y=percentage, fill=status_type)) +  # Corregido: era "dependents"
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
  filter(loan_status == " Rejected" & education == " Not Graduate")  # Nota el espacio antes de los valores

# Crear un dataframe con estos clientes pero cambiando su educación a 'Graduate'
modified_clients <- rejected_clients
modified_clients$education <- " Graduate"  # Nota el espacio antes de "Graduate"

# Predecir con el modelo para estos clientes modificados
modified_predictions <- predict(best_tree, modified_clients, type = "class")

# Identificar aquellos que serían aprobados con el cambio
would_be_approved <- which(modified_predictions == " Approved")  # Nota el espacio antes de "Approved"
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
  filter(loan_status == " Rejected")  # Nota el espacio antes de "Rejected"

# Función para encontrar ingresos mínimos necesarios para aprobación
find_min_income <- function(client, model, step = 1000000, max_increase = 10000000) {  # Ajustado paso y máximo
  # Copiar el cliente original
  test_client <- client
  original_income <- client$income_annum
  
  # Incrementar ingresos hasta que sea aprobado o se alcance el máximo
  for(income_increase in seq(0, max_increase, by = step)) {
    test_client$income_annum <- original_income + income_increase
    prediction <- predict(model, test_client, type = "class")
    
    if(prediction == " Approved") {  # Nota el espacio antes de "Approved"
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
  client_id = sampled_clients$loan_id,  # Usar el ID real del préstamo
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
  filter(loan_status == " Approved")  # Nota el espacio antes de "Approved"

# Función para encontrar el préstamo máximo que sería aprobado
find_max_loan <- function(client, model, step = 1000000, max_increase = 20000000) {  # Ajustado paso y máximo
  # Copiar el cliente original
  test_client <- client
  original_loan <- client$loan_amount
  
  # Incrementar monto del préstamo hasta que sea rechazado o se alcance el máximo
  last_approved <- original_loan
  
  for(loan_increase in seq(step, max_increase, by = step)) {
    test_client$loan_amount <- original_loan + loan_increase
    prediction <- predict(model, test_client, type = "class")
    
    if(prediction == " Rejected") {  # Nota el espacio antes de "Rejected"
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
  client_id = sampled_approved$loan_id,  # Usar el ID real del préstamo
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

# Resumen final
cat("\n-------------------------------------------------------------\n")
cat("RESUMEN DE RESULTADOS:\n")
cat("-------------------------------------------------------------\n")
cat(paste("1. Se generó un gráfico comparando porcentajes de aprobación reales vs predichos por número de dependientes.\n"))
cat(paste("2. Se identificaron", nrow(clients_education_impact), "clientes que serían aprobados si cambiaran su educación a 'Graduate'.\n"))
cat(paste("3. Para los clientes rechazados, el aumento promedio de ingresos necesario es de", 
          round(mean(income_results$percentage_increase, na.rm=TRUE), 2), "%.\n"))
cat(paste("4. Los clientes con préstamos aprobados podrían haber solicitado en promedio un", 
          round(mean(loan_results$percentage_increase, na.rm=TRUE), 2), "% más de monto.\n"))
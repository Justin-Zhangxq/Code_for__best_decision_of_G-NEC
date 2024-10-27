# Core data manipulation and visualization
library(tidyverse)  # Includes ggplot2, readr, and other data manipulation packages
library(gridExtra)

# Survival analysis
library(survival)
library(ggrisk)

# Machine learning and model fitting
library(glmnet)
library(mlr3)
library(mlr3verse)
library(mlr3proba)
library(mlr3extralearners)
library(distr6)

# Table creation
library(table1)

# Development tools
library(devtools)
library(usethis)

# Parallel processing
library(future)

# Uncomment the following lines if you need to install these packages
# devtools::install_github("mlr-org/mlr3proba")
# devtools::install_github("mlr-org/mlr3extralearners@*release")

source("tidyfuncs4sa.R")

set.seed(1986)

merged_data <- read_csv("your_files.csv")
str(merged_data)
factor_vars <- c(5, 6, 8:17, 19:28, 43:46, 48:50, 52, 53, 58, 61)

merged_data <- merged_data %>%
  mutate(across(all_of(factor_vars), as.factor))

datasplit <- rsample::initial_split(merged_data, prop = 0.7, strata = DFS)
train <- rsample::training(datasplit)
test <- rsample::testing(datasplit)

analysis_data <- merged_data[, -(1:3)]

sample_count <- nrow(analysis_data)
variable_count <- ncol(analysis_data)

cat(sprintf("Initial analysis includes %d patients with %d variables.\n", sample_count, variable_count))

x <- as.matrix(train[, -(1:3)])
y <- with(train, Surv(DFS_time, DFS))

fit <- glmnet(x, y, family = "cox")
cv_fit <- cv.glmnet(x, y, family = "cox", nfolds = 10)

par(mfrow = c(1, 1))
plot(fit, xvar = "lambda", label = TRUE)
plot(cv_fit, main = "")
plot(fit, xvar = "norm", label = TRUE)
plot(fit, xvar = "dev", label = TRUE)

cv_fit$lambda.min
coef_min <- coef(fit, s = cv_fit$lambda.min)
print(coef_min)

cat("Variables selected using lambda.min:\n")
selected_vars <- rownames(coef_min)[which(coef_min != 0)]
print(selected_vars)

# Select needed variables
selected_columns <- c("DFS_time", "DFS", selected_vars)

# Extract training and test sets with selected variables
train_selected <- rsample::training(datasplit) %>%
  select(all_of(selected_columns)) %>%
  sample_n(nrow(.))
test_selected <- rsample::testing(datasplit) %>%
  select(all_of(selected_columns)) %>%
  sample_n(nrow(.))

# Combine datasets
sadata2 <- rbind(train_selected, test_selected)

# Add 'set' column to sadata2
sadata2$set <- ifelse(row.names(sadata2) %in% row.names(train_selected), "Train", "Test")

# Convert 'set' column to factor
sadata2$set <- factor(sadata2$set, levels = c("Train", "Test"))

# Print sample counts
train_count <- sum(sadata2$set == "Train")
test_count <- sum(sadata2$set == "Test")
cat("Training set sample count:", train_count, "\n")
cat("Test set sample count:", test_count, "\n")
cat("Total sample count:", nrow(sadata2), "\n")

# Original model evaluation (kept for comparison)
x_test <- as.matrix(test[, -(1:3)])
test$pred_min <- predict(fit, newx = x_test, s = cv_fit$lambda.min)

model_min <- coxph(Surv(DFS_time, DFS) ~ pred_min, data = test)
cat("\nC-index (lambda.min):", summary(model_min)$concordance[1], "\n")


# Define time points of interest
itps <- c(12, 36, 60)
measure_sa <- msrs("surv.cindex")
# Create task objects
task_train <- as_task_surv(train_selected, time = "DFS_time", event = "DFS", type = "right")
task_test <- as_task_surv(test_selected, time = "DFS_time", event = "DFS", type = "right")

# Set up Random Survival Forest model
learner_rsf <- lrn(
  "surv.rfsrc",
  ntree = to_tune(100, 1000),
  mtry = to_tune(2, ceiling(sqrt(ncol(train_selected)))),
  nodesize = to_tune(5, 25)
)
learner_rsf$id <- "rsf"

# Set up parallel processing
future::plan("multisession")

# Hyperparameter tuning
tune_rsf <- tune(
  tuner = tnr("grid_search", resolution = 10),
  task = task_train,
  learner = learner_rsf,
  resampling = rsmp("cv", folds = 5),
  measure = msr("surv.cindex"),
  terminator = trm("none")
)
print(tune_rsf)

# Visualize hyperparameter tuning results
as.data.table(tune_rsf$archive) %>%
  as.data.frame() %>%
  select(1:4) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~surv.cindex, 
                colorscale = 'Jet', 
                showscale = T),
    dimensions = list(
      list(label = 'ntree', values = ~ntree),
      list(label = 'mtry', values = ~mtry),
      list(label = 'nodesize', values = ~nodesize)
    )
  ) %>%
  plotly::layout(title = "RSF HPO Guided by C-Index",
                 font = list(family = "serif"))

# Train final model with optimized hyperparameters
learner_rsf$param_set$values <- tune_rsf$result_learner_param_vals
learner_rsf$train(task_train)

# Prepare merged data for prediction
merged_data_selected <- merged_data[, c("DFS_time", "DFS", selected_vars)]
pred_merged_data <- learner_rsf$predict_newdata(merged_data_selected)
predprob_merged_data <- predprob(pred_merged_data, merged_data_selected, "DFS_time", "DFS", "rsf", "all", itps)

# Predict on training set
predtrain_rsf <- learner_rsf$predict(task_train)
predprobtrain_rsf <- predprob(
  pred = predtrain_rsf, 
  preddata = train_selected, 
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf", 
  dataset = "train_selected", 
  timepoints = itps
)

# Performance metrics for training set
predtrain_rsf$score(measure_sa)
cindex_bootci(learner_rsf, train_selected)

library(mets)

# Evaluate training set predictions
evaltrain_rsf <- eval4sa(
  predprob = predprobtrain_rsf,
  preddata = train_selected,
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf",
  dataset = "train_selected",
  timepoints = itps,
  plotcalimethod = "quantile",
  bw4nne = NULL,
  q4quantile = 5,
  cutoff = "median"
)

# Display evaluation results for training set
print(evaltrain_rsf$auc)
print(evaltrain_rsf$rocplot)
print(evaltrain_rsf$brierscore)
print(evaltrain_rsf$brierscoretest)
print(evaltrain_rsf$calibrationplot)
print(evaltrain_rsf$riskplot)

# Decision curve analysis for training set
sadca(
  predprob = predprobtrain_rsf,
  preddata = train_selected,
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf",
  dataset = "train",
  timepoints = itps,
  timepoint = 12, 
  xrange = 0:100 / 100
)

# Predict on test set
predtest_rsf <- learner_rsf$predict(task_test)
predprobtest_rsf <- predprob(
  pred = predtest_rsf, 
  preddata = test_selected, 
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf", 
  dataset = "test", 
  timepoints = itps
)

# Performance metrics for test set
predtest_rsf$score(measure_sa)
cindex_bootci(learner_rsf, test_selected)

# Evaluate test set predictions
evaltest_rsf <- eval4sa(
  predprob = predprobtest_rsf,
  preddata = test_selected,
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf",
  dataset = "test",
  timepoints = itps,
  plotcalimethod = "quantile",
  bw4nne = NULL,
  q4quantile = 5,
  cutoff = "median"
)

# Display evaluation results for test set
print(evaltest_rsf$auc)
print(evaltest_rsf$rocplot)
print(evaltest_rsf$brierscore)
print(evaltest_rsf$brierscoretest)
print(evaltest_rsf$calibrationplot)
print(evaltest_rsf$riskplot)

# Decision curve analysis for test set
sadca(
  predprob = predprobtest_rsf,
  preddata = test_selected,
  etime = "DFS_time",
  estatus = "DFS",
  model = "rsf",
  dataset = "test",
  timepoints = itps,
  timepoint = 12, 
  xrange = 0:100 / 100
)

# Step 1: Predict on all data
merged_data_selected <- merged_data[, c("DFS_time", "DFS", selected_vars)]
pred_merged_data <- learner_rsf$predict_newdata(merged_data_selected)
predprob_merged_data <- predprob(pred_merged_data, merged_data_selected, "DFS_time", "DFS", "rsf", "all", itps)

# Step 2: Merge prediction results
result_data <- cbind(merged_data, predprob_merged_data[, as.character(itps)])
new_names <- paste0("DFS_prob_", itps, "_months")
colnames(result_data)[(ncol(merged_data) + 1):ncol(result_data)] <- new_names

# Print column names for verification
print("Columns in result_data:")
print(colnames(result_data))

# Print DFS probability column names for each chemotherapy regimen
for (month in itps) {
  print(paste0("DFS probability columns for ", month, " months:"))
  print(paste0("DFS_prob_", month, "_months_chemo", 1:3))
}

# Step 3: Predict results for different chemotherapy regimens
if ("Chemo_regimen" %in% selected_vars) {
  chemo_levels <- levels(merged_data$Chemo_regimen)
  
  for (i in seq_along(chemo_levels)) {
    data_copy <- merged_data_selected
    data_copy$Chemo_regimen[] <- factor(chemo_levels[i], levels = chemo_levels)
    pred <- learner_rsf$predict_newdata(data_copy)
    
    if (!exists("predprob")) {
      stop("predprob function is not defined. Please check your code.")
    }
    
    predprob_result <- predprob(pred, data_copy, "DFS_time", "DFS", "rsf", "all", itps)
    new_names <- paste0("DFS_prob_", itps, "_months_chemo", i)
    result_data <- cbind(result_data, setNames(predprob_result[, as.character(itps)], new_names))
  }
  
  # Step 4: Identify the optimal treatment strategy
  for (month in itps) {
    col_name <- paste0("optimal_treatment_strategies_", month, "_months")
    result_data[[col_name]] <- apply(result_data[, paste0("DFS_prob_", month, "_months_chemo", 1:3)], 1, function(x) chemo_levels[which.max(x)])
  }
  
  # Step 5: Create a table of optimal treatment strategies
  optimal_strategies_table <- table1(
    ~ optimal_treatment_strategies_12_months + 
      optimal_treatment_strategies_36_months + 
      optimal_treatment_strategies_60_months, 
    data = result_data
  )
  
  print("Optimal Treatment Strategies Table:")
  print(optimal_strategies_table)
  
  # Additional summary statistics
  for (month in itps) {
    col_name <- paste0("optimal_treatment_strategies_", month, "_months")
    cat("\nOptimal Treatment Strategies for", month, "months:\n")
    print(table(result_data[[col_name]]))
    cat("\n")
  }
}

# Close parallel processing
future::plan("sequential")

# Print summary of result_data
cat("\nSummary of result_data:\n")
print(summary(result_data))

# Scott Withrow

# Random Forest Analysis

# This is a follow-up to the Naive Bayes example. Here, we replicate the analysis done there but use a random forest decision
# tree to predict performance category.

# Packages -------------------------------------------------------------------------------------------------------------

# Load required packages
require(tidyverse)
require(tidymodels)
require(rpart.plot)
require(vip)

# Data -----------------------------------------------------------------------------------------------------------------

# Read in dataset
data_frame_full <- read_rds("")

# The only real data processing step here is to ensure our performance category is a factor
data_frame_full$Performance_Category <- factor(data_frame_full$Performance_Category)

# Since we will be invoking random numbers and want these examples to be reproducible we will set a seed.
set.seed(328)

# Next, let's pull out a low n testing set. We will use sample_n to find 25 observations. Then, filter those observations out
# of the original data.
low_n_testing <- sample_n(data_frame_full, 25)
data_frame_full <- filter(data_frame_full, !(ID %in% low_n_testing$ID))

# Next, we will pull out a large n testing set of 200 observations.
large_n_testing <- sample_n(data_frame_full, 200)
data_frame_full <- filter(data_frame_full, !(ID %in% large_n_testing$ID))

# Finally, we will split the remaining data into a training and a testing set. We will use initial_split to pick out 80% of
# data into the training set and 20% into the testing set. We will use strata to ensure that roughly the same proportion of
# high/average/low performance categories in each sample.
data_split <- initial_split(data_frame_full,
                            prop = 0.8,
                            strata = Performance_Category)

train_data <- training(data_split)
test_data <- testing(data_split)

# We can then drop our original dataframe
rm(data_frame_full)

# Random Forest Decision Tree ------------------------------------------------------------------------------------------

# Set up the model
rf_model <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# Specify a tuning grid
rf_tune_grid <- 
  grid_regular(cost_complexity(),
               tree_depth(),
               levels = 5)

# Set up the recipe
rf_recipe <-
  recipe(Performance_Category ~ ., data = train_data) %>%
  update_role(ID, new_role = "ID") %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Define cross-validation
folds <- vfold_cv(train_data, v = 10)

# Create a workflow
rf_workflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# Compute fit
rf_fit <-
  rf_workflow %>%
  tune_grid(resamples = folds, grid = rf_tune_grid)

# View fit
collect_metrics(rf_fit)

# Find the most accurate model
rf_fit %>%
  show_best("accuracy")

best_tree <- rf_fit %>%
  select_best("accuracy")

# Graph out the accuracy for the tree depth and cost complexity to find the sweet spot with the most penalization that
# doesn't reduce accuracy.
rf_fit %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

# Select the most accurate model
final_wf <- 
  rf_workflow %>% 
  finalize_workflow(best_tree)

# Compute final fit. We run this against all the data.
final_fit <- 
  final_wf %>%
  last_fit(data_split) 


# Evaluation and Prediction --------------------------------------------------------------------------------------------

# Pull out the decision tree information
final_tree <- extract_workflow(final_fit)

# Visualize the tree
# Warning. This tree is massive and depending on your machine specs may crash your R session.
final_tree %>%
  extract_fit_engine() %>%
  rpart.plot(roundint = FALSE)

# Show which features are the most important
final_tree %>% 
  extract_fit_parsnip() %>% 
  vip()

# Now, try and use some of what you learned from the Naive Bayes example to complete this exercise. We can fit the random
# forest decision tree to the small n and large n samples and see how well we can predict job performance with a random
# forest versus LDA or Naive Bayes. Similarly, we could build all sorts of machine learning models with these same data!
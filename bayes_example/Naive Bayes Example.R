# Scott Withrow

# Bayes Classification

# Classify into 3 performance buckets using 26 indicator variables. If the below packages are not installed they will need to
# be in order to proceed.

# The goal of this script is to demonstrate how to use the tidy framework for machine learning with a simple but applicable
# example. We will run a Naive Bayesian Classification model, test it versus a holdback sample, then show how it would work
# relative to a similar analysis done in a "traditional" fashion.


# Packages -------------------------------------------------------------------------------------------------------------

# Load required packages
require(tidyverse) # Data manipulation
require(tidymodels) # Machine Learning Modeling with intuitive interface
require(discrim) # Contains the naive_Bayes model we will use
require(klaR) # Contains the engine for naive_Bayes


# Data -----------------------------------------------------------------------------------------------------------------

# Read in dataset
data_frame_full <- read_rds("C:/Users/degov/Google Drive/SIOP 2022 ML Seminar/scottscrap/Data.RDS")

# The only real data processing step here is to ensure our performance category is a factor
factor(data_frame_full$Performance_Category) -> data_frame_full$Performance_Category

# Split data into Training and Testing sets. The goal here is to take a majority of data and put it into a training dataset.
# We will use some re-sampling in order to try and establish a good statistical model without over-fitting but we will need to
# have a "hold-back" or testing set to verify that the model works as intended. We will also hold out two samples that are
# more representative of the kinds of data a client may provide in order to show how these things may work.

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


# Training -------------------------------------------------------------------------------------------------------------

# In this section we will train our naive Bayes model. To do this we will first set up a "recipe" where we specify the
# outcome variable and the input features. We will use all the features in the dataframe. However, we do need to make sure
# the recipe is aware that the ID variable is in the dataset. Finally, we will center and scale all the predictors. This
# isn't really necessary with this model but it helps to illustrate how recipes are formed of many steps.
naive_bayes_recipe <-
  recipe(Performance_Category ~ ., data = train_data) %>% # Formula statement
  update_role(ID, new_role = "ID") %>% # Identify ID variable as an ID variable
  step_zv(all_predictors()) %>% # Remove zero variance predictors
  step_normalize(all_predictors()) # Normalize

# Depending on the types of data in the set you may need to have extra steps here to specify how a date variable would be
# read, which variables would be dummy coded, etc. However, we just have a bunch of continuous data so we don't need to worry
# about any additional steps to the recipe statement.

# Next we want to specify a resampling strategy to ensure our modeling process doesn't overfit. In this case we will use a
# simple cross-validation strategy.
folds <- vfold_cv(train_data, v = 10)

# Next, we specify how the model will work. We will use naive_Bayes from the parsnip package which ties in with the discrim
# package for classification models. We will also need to specify an engine which comes from the klaR package. See
# ?details_naive_Bayes_klaR for a bit more information on these packages.
naive_bayes_model <- 
  naive_Bayes() %>% # use naive Bayes
  set_mode("classification") %>% # Output is a classification (rather than regression)
  set_engine("klaR") %>% # Engine to use is klaR
  set_args(usekernel = FALSE) # We can assume all our predictors are normally distributed

# Now that we know what the model looks like and how we are going to compute that model we can do some actual modeling. It
# should be noted that at this point we haven't really run anything, we have just been setting up HOW things should be run.
# These kinds of recipes and model statements are very flexible and allow for many different kinds of models to be run in a
# fairly simple manner. tidymodels also simplifies the output and graphing a bit as well.

# For processing we will use a workflow. A workflow joins together a model and a recipe (as well as other things we won't use
# here). A workflow is powerful as it allows us to run the same model against different sets of data with very little
# additional code work.
naive_bayes_workflow <-
  workflow() %>%
  add_model(naive_bayes_model) %>%
  add_recipe(naive_bayes_recipe)

# Now we can compute a fit. It is at this point that we actually run the model. Depending on your hardware this may take some
# time.
naive_bayes_fit <-
  naive_bayes_workflow %>%
  fit_resamples(folds)

# Print out the fit information into the console. Warning: lots of data will be spit out into the console. 
naive_bayes_fit

# We can examine the fit metrics for the 10 models
collect_metrics(naive_bayes_fit)

# Or extract the "best" model
naive_bayes_fit %>%
  show_best("accuracy")

# Once we know what the "best" model is we can run it over the full training data. This should be familiar to you from the
# above code. The first step is we take the workflow from above and specify which model to run. In this case, we didn't
# really "tune" any of our parameters so there isn't a huge value from this. However, if we were to say tune smoothness in
# our model to adjust how flexible class boundaries are this would have greater value.
final_naive_bayes_workflow <-
  naive_bayes_workflow %>%
  finalize_workflow(select_best(naive_bayes_fit, "accuracy"))
final_naive_bayes_fit <-
  final_naive_bayes_workflow %>%
  fit(data = train_data)

# Evaluation and Prediction --------------------------------------------------------------------------------------------

# For evaluation purposes we first need to "augment" our data. This uses the fit model to generate new classes as
# probabilities and appends them to our dataset.
augment(final_naive_bayes_fit, new_data = test_data) -> naive_bayes_augmented

# We can pull in a confusion matrix in order to see how well we predict in our testing dataset
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# We can also assess the accuracy of the classification
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# Overall, our accuracy in prediction is around 0.468 which isn't that great. Perhaps we should try other models?


# Naive Bayes vs Linear Discriminant Analysis in Small n Samples -------------------------------------------------------

# For a practical example comparing the two let's look at what would normally happen in an applied setting. Typically, given
# this setup the strategy would be to fit a Linear Discriminant Analysis to find weights that predict classification. So,
# let's compute that model and compare it to our Bayes model above.

# First, we would need to do a training and a testing sample (even with low n!)
data_split <- initial_split(low_n_testing,
                            prop = 0.8,
                            strata = Performance_Category)

train_data <- training(data_split)
test_data <- testing(data_split)

# Then we would use lda from the MASS package to fit the model. We will use the same tidymodels framework as above for
# simplicity.
lda_recipe <-
  recipe(Performance_Category ~ ., data = train_data) %>%
  update_role(ID, new_role = "ID") %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

lda_model <- 
  discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS") %>%
  set_args(usekernel = FALSE)

lda_workflow <-
  workflow() %>%
  add_model(lda_model) %>%
  add_recipe(lda_recipe)

lda_fit <-
  lda_workflow %>%
  fit(data = train_data)

# Check how well it predicts the class membership
augment(lda_fit, new_data = test_data) -> lda_augmented
conf_mat(lda_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(lda_augmented, truth = Performance_Category, estimate = .pred_class)

# Now let's compare that to our Bayes. Because we already fit the model we just need to predict and can use the full dataset!
augment(final_naive_bayes_fit, new_data = low_n_testing) -> naive_bayes_augmented
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# But, Let's be fair and check it versus the smaller holdback sample
augment(final_naive_bayes_fit, new_data = test_data) -> naive_bayes_augmented
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# As we can see here LDA made a couple of mistakes in the classification while Bayes' did not (with the help of that prior)


# Naive Bayes vs Linear Discriminant Analysis in Large  n Samples ------------------------------------------------------

# We basically reproduce the exact same analysis as above with the dataset of 200

# First, we would need to do a training and a testing sample (even with low n!)
data_split <- initial_split(large_n_testing,
                            prop = 0.8,
                            strata = Performance_Category)

train_data <- training(data_split)
test_data <- testing(data_split)

# Then we would use lda from the MASS package to fit the model. We will use the same tidymodels framework as above for
# simplicity.
lda_recipe <-
  recipe(Performance_Category ~ ., data = train_data) %>%
  update_role(ID, new_role = "ID") %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

lda_model <- 
  discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS") %>%
  set_args(usekernel = FALSE)

lda_workflow <-
  workflow() %>%
  add_model(lda_model) %>%
  add_recipe(lda_recipe)

lda_fit <-
  lda_workflow %>%
  fit(data = train_data)

# Check how well it predicts the class membership
augment(lda_fit, new_data = test_data) -> lda_augmented
conf_mat(lda_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(lda_augmented, truth = Performance_Category, estimate = .pred_class)

# Now let's compare that to our Bayes. Because we already fit the model we just need to predict and can use the full dataset!
augment(final_naive_bayes_fit, new_data = large_n_testing) -> naive_bayes_augmented
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# But, Let's be fair and check it versus the smaller holdback sample
augment(final_naive_bayes_fit, new_data = test_data) -> naive_bayes_augmented
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# As we can see here LDA made a couple of mistakes in the classification while Bayes' was much more accurate. 


# Follow-up Questions and Practice -------------------------------------------------------------------------------------

# It's clear from the above analyses that neither Bayes' or LDA were particularly great at predicting performance from our
# features. But, those features are rooted in personality and general cognitive ability so the low accuracy is to be
# expected. What is encouraging in that in situations where sample size may be low the prior information we have from the
# Bayes framework can make prediction better. Plus, it works better than treating every study as it's own unique analytic.
# However, there may be better models out there for these classification problems!

# 1) Is Naive Bayes the correct approach to take here? Try fitting a random forest model and see if you can increase
# prediction. Hint, rand_forest() %>% set_engine("ranger") %>% set_mode("classification") to get you started.

# 2) We absolutely violated the independent predictor assumption of this model. Are there other models we could use or
# parameters we could change in this model to help it fit better?

# 3) One element of machine learning we didn't explore here is "tuning" a model. Tuning a model consists of adjusting the
# hyperparameters of a model (the nuts and bolts under the hood) to get better prediction. tidymodels and the parsnip package
# make that fairly easy to accomplish with the tune() function. For example:

naive_bayes_recipe <-
  recipe(Performance_Category ~ ., data = train_data) %>%
  update_role(ID, new_role = "ID") %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

folds <- vfold_cv(train_data, v = 10)

naive_bayes_model <-
  naive_Bayes(smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("klaR") %>%
  set_args(usekernel = FALSE)

naive_bayes_tuning <- 
  grid_regular(smoothness(), levels = 10)

naive_bayes_workflow <-
  workflow() %>%
  add_model(naive_bayes_model) %>%
  add_recipe(naive_bayes_recipe)

naive_bayes_fit <-
  naive_bayes_workflow %>%
  tune_grid(resamples = folds,
            grid = naive_bayes_tuning)

# We can examine the fit metrics for the models
collect_metrics(naive_bayes_fit)

# Or extract the "best" model
naive_bayes_fit %>%
  show_best("accuracy")

# Check for fit versus our training data
final_naive_bayes_workflow <-
  naive_bayes_workflow %>%
  finalize_workflow(select_best(naive_bayes_fit, "accuracy"))
final_naive_bayes_fit <-
  final_naive_bayes_workflow %>%
  fit(data = train_data)

augment(final_naive_bayes_fit, new_data = test_data) -> naive_bayes_augmented
conf_mat(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)
accuracy(naive_bayes_augmented, truth = Performance_Category, estimate = .pred_class)

# In this case did we enhance prediction by tuning our parameters? 
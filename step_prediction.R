suppressMessages({
  library(data.table)
  library(RcppRoll)
  library(dplyr)
  library(lightgbm)
})

# Garbage collection
igc <- function() {
  invisible(gc()); invisible(gc())   
}
igc()
path <- "data/"


calendar <- fread(file.path(path, "calendar.csv"))
selling_prices <- fread(file.path(path, "sell_prices.csv"))
sample_submission <- fread(file.path(path, "sample_submission.csv"))
sales <- fread(file.path(path, "sales_train_validation.csv"))


data_prerocess <- function(calendar, selling_prices, sample_submission, sales, i) {
  
  # Calendar
  calendar[, `:=`(date = NULL, 
                  weekday = NULL, 
                  d = as.integer(substring(d, 3)))]
  cols <- c("event_name_1", "event_type_1")#, "event_name_2", "event_type_2")
  calendar[, (cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cols]
  
  # Selling prices                   
  selling_prices[, `:=`(
    sell_price_rel_diff = sell_price / dplyr::lag(sell_price) - 1,
    sell_price_cumrel = (sell_price - cummin(sell_price)) / (1 + cummax(sell_price) - cummin(sell_price)),
    sell_price_roll_sd7 = roll_sdr(sell_price, n = 7)
  ), by = c("store_id", "item_id")]
  
  # Sales: Reshape
  sales[, id := gsub("_validation", "", id)]                         
  empty_dt = matrix(NA_integer_, ncol = 2 * 28, nrow = 1, dimnames = 
                      list(NULL, paste("d", 1913 + 1:(2 * 28), sep = "_")))
  
  sales <- cbind(sales, empty_dt)
  sales <- melt(sales, id.vars = c("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"), 
                variable.name = "d", value = "demand")
  sales[, d := as.integer(substring(d, 3))]
  
  # Sales: Reduce size
  sales <- sales[d >= 1000]
  
  # Sales: Feature construction: Subset of features from very fst kernel
  stopifnot(!is.unsorted(sales$d))
  sales[, lag_t28 := dplyr::lag(demand, i), by = "id"]
  sales[, `:=`(rolling_mean_t7 = roll_meanr(lag_t28, 7),
               rolling_mean_t30 = roll_meanr(lag_t28, 30),
               rolling_mean_t60 = roll_meanr(lag_t28, 60),
               rolling_mean_t90 = roll_meanr(lag_t28, 90),
               rolling_mean_t180 = roll_meanr(lag_t28, 180),
               rolling_sd_t7 = roll_sdr(lag_t28, 7),
               rolling_sd_t30 = roll_sdr(lag_t28, 30)), 
        by = "id"]
  igc()
  
  sales <- sales[d >= 1914 | !is.na(rolling_mean_t180)]
  
  ## Merge calendar to sales
  sales <- calendar[sales, on = "d"]
  #igc()
  
  # Merge selling prices to sales and drop key
  train <- selling_prices[sales, on = c('store_id', 'item_id', 'wm_yr_wk')][, wm_yr_wk := NULL]
  rm(sales, selling_prices, calendar)
  igc()
  
  # Turn non-numerics to integer
  cols <- c("item_id", "dept_id", "cat_id", "store_id", "state_id")
  train[, (cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cols]
  
  # Covariables used
  x <- c("wday", "month", "year", 
         "event_name_1", "event_type_1", #"event_name_2", "event_type_2", 
         "snap_CA", "snap_TX", "snap_WI",
         "sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7",
         "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", 
         "rolling_mean_t90", "rolling_mean_t180", "rolling_sd_t7", "rolling_sd_t30",
         "item_id", "dept_id", "cat_id", "store_id", "state_id")
  
  # Separate submission data and reconstruct id columns
  test <- train[d >= 1914]
  test[, id := paste(id, ifelse(d <= 1941, "validation", "evaluation"), sep = "_")]
  test[, F := paste0("F", d - 1913 - 28 * (d > 1941))]
  
  # 1 month of validation data
  flag <- train$d < 1914 & train$d >= 1914 - 28
  valid <- lgb.Dataset(data.matrix(train[flag, x, with = FALSE]), 
                       label = train[["demand"]][flag])
  
  # Final preparation of training data
  flag <- train$d < 1914 - 28
  y <- train[["demand"]][flag]
  train <- data.matrix(train[flag, x, with = FALSE])
  igc()
  train <- lgb.Dataset(train, label = y)
  igc()
  
  # Parameter choice
  params = list(objective = "poisson",
                metric = "rmse",
                seed = 20,
                learning_rate = 0.1,
                alpha = 0.1,
                lambda = 0.1,
                num_leaves = 63,
                bagging_fraction = 0.66,
                bagging_freq = 2, 
                colsample_bytree = 0.77)
  
  fit <- lgb.train(params, train, num_boost_round = 2000, 
                   eval_freq = 100, early_stopping_rounds = 200, 
                   valids = list(valid = valid))
  
  test <- test[F == paste0('F', i)]
  pred <- predict(fit, data.matrix(test[, x, with = FALSE]))
  test[, demand := pmax(0, pred)]
  test_long <- dcast(test, id ~ F, value.var = "demand")
  
  return(test_long)
}

submit_data <- list()

for(i in 1:28){
  model_data <- data_prerocess(calendar, selling_prices, sample_submission, sales, i)
  submit_data[[i]] <- model_data
  print(i)
  igc()
}


library(data.table)
library(lightgbm)

set.seed(890510)

h <- 28 
max_lags <- 400
tr_last <- 1913
fday <- as.IDate("2016-04-25") 



#---------------------------
cat("Creating auxiliary functions...\n")

free <- function() invisible(gc())

is_train = TRUE
nrows = Inf

stats = dt[, c(1:6, 1526:1919)]


create_dt <- function(is_train = TRUE, nrows = Inf) {
  
  prices <- fread("data/sell_prices.csv")
  cal <- fread("data/calendar.csv", drop = "weekday")
  cal[, date := as.IDate(date, format="%Y-%m-%d")]
  
  if (is_train) {
    dt <- fread("data/sales_train_validation.csv", nrows = nrows)
    dt <- dt[, c(1:6, 1526:1919)]
  } else {
    dt <- fread("data/sales_train_validation.csv", nrows = nrows,
                drop = paste0("d_", 1:(tr_last-max_lags)))
    dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := NA_real_]
  }
  
  dt <- melt(dt,
             measure.vars = patterns("^d_"),
             variable.name = "d",
             value.name = "sales")
  
  dt <- dt[cal, `:=`(date = i.date, 
                     wm_yr_wk = i.wm_yr_wk,
                     event_name_1 = i.event_name_1,
                     snap_CA = i.snap_CA,
                     snap_TX = i.snap_TX,
                     snap_WI = i.snap_WI), on = "d"]
  
  dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")]
}

create_fea <- function(dt) {
  
  lag <- c(7, 28, 29)
  dt[, (paste0("lag_", lag)) := shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  win <- c(7, 30, 90, 180)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win), by = "id"]
  
  win <- c(7, 30)
  dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]
  
  dt[, price_change_1 := sell_price / shift(sell_price) - 1, by = "id"]
  dt[, price_change_365 := sell_price / frollapply(shift(sell_price), 365, max) - 1, by = "id"]
  
  cols <- c("item_id", "state_id", "dept_id", "cat_id", "event_name_1")   
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols]
  
  dt[, `:=`(wday = wday(date),
            mday = mday(date),
            week = week(date),
            month = month(date),
            year = year(date),
            store_id = NULL,
            d = NULL,
            wm_yr_wk = NULL)]
}

#---------------------------
cat("Creating training set with features...\n")

tr <- create_dt()

free()

create_fea(tr)
free()

tr <- na.omit(tr)
y <- tr$sales

idx <- tr[date <= max(date)-h, which = TRUE] 

tr[, c("id", "sales", "date") := NULL]
free()

tr <- data.matrix(tr)
free()

#---------------------------
cat("Constructing training and validation sets for GBM...\n")

cats <- c("item_id", "state_id", "dept_id", "cat_id",
          "wday", "mday", "week", "month", "year",  
          "snap_CA", "snap_TX", "snap_WI")

xtr <- lgb.Dataset(tr[idx, ], label = y[idx], categorical_feature = cats)
xval <- lgb.Dataset(tr[-idx, ], label = y[-idx], categorical_feature = cats)

rm(tr, y, idx)
free()

#---------------------------
cat("Training model...\n")

p <- list(objective = "regression_l2",
          metric ="rmse",
          learning_rate = 0.05,
          sub_feature = 0.75,
          sub_row = 0.75,
          bagging_freq = 10,
          lambda = 0.1,
          alpha = 0.1,
          nthread = 16)

m_lgb <- lgb.train(params = p,
                   data = xtr,
                   nrounds = 2000,
                   valids = list(train = xtr, valid = xval),
                   early_stopping_rounds = 200,
                   eval_freq = 10,
                   verbose = 0)

lgb.plot.importance(lgb.importance(m_lgb), 20)

rm(xtr, xval, p)
free()

#---------------------------
cat("Generating predictions...\n")

te <- create_dt(FALSE)

for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea(tst)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := predict(m_lgb, tst)]
}

te[date >= fday
   ][date >= fday+h, id := sub("validation", "evaluation", id)
     ][, d := paste0("F", 1:28), by = id
       ][, dcast(.SD, id ~ d, value.var = "sales")
         ][, fwrite(.SD, "data/sub_dt_lgb.csv")]

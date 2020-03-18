library(tidyverse)
library(data.table)
library(lightgbm)

set.seed(890510)

h <- 28 
max_lags <- 400
tr_last <- 1913
fday <- as.IDate("2016-04-25") #예측값 생성

free <- function() invisible(gc())

#---------------------------
cat("Creating auxiliary functions...\n")


## 데이터 불러오기
prices <- fread("data/sell_prices.csv")
cal <- fread("data/calendar.csv", drop = "weekday")
cal[, date := as.IDate(date, format="%Y-%m-%d")]
dt <- fread("data/sales_train_validation.csv")

##분석을 위해 데이터를 wide form 을 long form으로
dt <- melt(dt, measure.vars = patterns("^d_"), variable.name = "d", value.name = "sales")
  
## calendar 데이터와, train 데이터를 join -> train 데이터에 날짜 정보 추가
dt <- dt[cal, `:=`(date = i.date
                   , wm_yr_wk = i.wm_yr_wk
                   , event_name_1 = i.event_name_1
                   , snap_CA = i.snap_CA
                   , snap_TX = i.snap_TX
                   , snap_WI = i.snap_WI), on = "d"]

## price 데이터와 train 데이터의 join -> 상품별 가격정보 추가  
dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")]

# lag 변수 추가
lag <- c(7, 14, 21, 28)
dt[, (paste0("lag_", lag)) := shift(.SD, lag), .SDcols = "sales", by = "id"]

#추세를 반영하기 위한 roll_mean변수 추가 1)평균, 2)최대값
win <- c(7, 28, 28*3, 28*6)
dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win), by = "id"]

win <- c(7, 28)
dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]

#가격 변동에 관한 변수
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

#---------------------------
cat("Creating training set with features...\n")

free()

dt <- na.omit(dt)
dt <- dt[date >= '2014-01-01']

y <- dt$sales
idx <- dt[date <= max(date)-h, which = TRUE] 

dt[, c("id", "sales", "date") := NULL]
free()

dt <- data.matrix(dt)
free()

#---------------------------
cat("Constructing training and validation sets for GBM...\n")

cats <- c("item_id", "state_id", "dept_id", "cat_id",
          "wday", "mday", "week", "month", "year",  
          "snap_CA", "snap_TX", "snap_WI")


xtr <- lgb.Dataset(dt[idx, ], label = y[idx], categorical_feature = cats)
xval <- lgb.Dataset(dt[-idx, ], label = y[-idx], categorical_feature = cats)
xdat <- lgb.Dataset(dt, label = y, categorical_feature = cats)

rm(dt)
#---------------------------
cat("Training model...\n")

p <- list(objective = "regression_l2",
          metric ="rmse",
          learning_rate = 0.05,
          sub_feature = 0.75,
          sub_row = 0.75,
          bagging_freq = 5,
          lambda = 0.1,
          alpha = 0.1,
          nthread = 10)

m_lgb <- lgb.train(params = p,
                   data = xtr,
                   nrounds = 4000,
                   valids = list(train = xtr, valid = xval),
                   early_stopping_rounds = 100,
                   eval_freq = 50)

lgb.plot.importance(lgb.importance(m_lgb), 20)

m_lgb <- lgb.train(params = p,
                   data = xdat,
                   nrounds = m_lgb$best_iter)

# 기존모형 [2000]:	train's rmse:2.06511	valid's rmse:1.90673 

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
         ][, fwrite(.SD, "data/few_feature_full_data.csv")]

#################################33 함수
create_dt <- function(is_train = TRUE, nrows = Inf) {
  
  prices <- fread("data/sell_prices.csv")
  cal <- fread("data/calendar.csv", drop = "weekday")
  cal[, date := as.IDate(date, format="%Y-%m-%d")]
  
  if (is_train) {
    dt <- fread("data/sales_train_validation.csv", nrows = nrows)
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
  
  lag <- c(7, 14, 21, 28)
  dt[, (paste0("lag_", lag)) := shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  #추세를 반영하기 위한 roll_mean변수 추가 1)평균, 2)최대값
  win <- c(7, 28, 28*3, 28*6)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win), by = "id"]
  
  win <- c(7, 28)
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

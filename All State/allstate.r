#loading packages
# install.packages("xgboost", repos=c("http://dmlc.ml/drat/", getOption("repos")), type="source")
rm(list=ls())
library(ggplot2)
library(xgboost)
library(data.table)
library(gridExtra)
library(e1071)
library(corrplot)
library(Matrix)
library(Metrics)

#Setting wd and loading data
#Setting wd and loading data
train.data = fread("AllState/train.csv")
train.data$train_flag = 1
test.data = fread("AllState/test.csv")
test.data$loss = 0
test.data$train_flag = 0
IDs = test.data$id
data = as.data.frame(rbind(train.data, test.data))
data$id = NULL

#Converting all character variables to numeric
for (i in names(data))  {
  if(class(data[[i]]) == "character")
           data[[i]] = as.numeric(as.factor(data[[i]]))
}       
    
#Separating test, development and validation data set
train_data = data[data$train_flag == 1,]
test_data = data[data$train_flag == 0,]
set.seed(1)
dev.rows = sample(1:nrow(train_data), 0.8*nrow(train_data), replace = F)
dev.data = train_data[dev.rows,]
valid.data = train_data[-dev.rows,]
dev.data$loss = log(dev.data$loss + 200)



# XGBOOST Model -----------------------------------------------------------

X_train = dev.data[,c(1:130)]
Y_train = dev.data$loss
X_valid = valid.data[,c(1:130)]
Y_valid = valid.data$loss
X_test = test_data[,c(1:130)]

#XGB Training
best_mae = Inf
best_mae_index = 0
shift = 200

for (iter in 1:2) {
  cat(paste0("iter= ", iter))
  param <- list(objective = "reg:linear",
                eval_metric = "mae",
                max_depth = sample(6:12, 1),
                eta = runif(1, .01, .1),
                gamma = runif(1, 0.0, 0.2),
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8),
                min_child_weight = sample(1:10, 1),
                max_delta_step = sample(1:10, 1)
  )
  print(unlist(param))
  cv.nround = 2000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=as.matrix(X_train), label=as.matrix(Y_train), params = param, nthread=6,  nfold = cv.nfold, nrounds = cv.nround, verbose = T, early_stopping_rounds = 10)

  min_mae = min(mdcv$evaluation_log[, test_mae_mean])
  min_mae_index = which.min(mdcv$evaluation_log[, test_mae_mean])

  if (min_mae < best_mae) {
    best_mae = min_mae
    best_mae_index = min_mae_index
    best_seednumber = seed.number
    best_param = param
  }
  
}
nround = best_mae_index
set.seed(best_seednumber)
# nround  1488
# set.seed(best_seednumber)
# best_seednumber  8155
# best_param 
# $objective  "reg:linear"
# $eval_metric  "mae"
# $max_depth 8
# $eta 0.01945826
# $gamma 0.1152051
# $subsample  0.7288189
# $colsample_bytree  0.7626341
# $min_child_weight 5
# $max_delta_step  1

model_xgb1 <- xgb.train(data = xgb.DMatrix( as.matrix(X_train), label = as.matrix(Y_train) ), params = best_param, nrounds = nround, verbose = T, nthread=6) 

#Validation data prediction
predict.xgbm <- exp(predict(model_xgb1, data.matrix(X_valid) )) - shift
mean(abs(predict.xgbm - Y_valid))

#training on entire dataset
model_xgb2 <- xgb.train(data = xgb.DMatrix(as.matrix(train_data[,1:130]), label = as.matrix(log(train_data$loss + shift))), params=best_param, nrounds = round(nround/0.8), verbose = T, nthread=6)

#making prediction and writing submission file
predict.xgbm <- exp(predict(model_xgb2, data.matrix(X_test))) - shift
sub_xgbm <- data.frame(id = IDs, loss = predict.xgbm)
write.csv(sub_xgbm, file = "sub_xgb.csv", row.names = F)


########################################### GBM in H2O #####################################

library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.clusterIsUp()

#data to h2o cluster
dev.h2o <- as.h2o(dev.data)
valid.h2o <- as.h2o(valid.data)
test.h2o <- as.h2o(test_data)

#check column index number
colnames(dev.h2o)
colnames(test.h2o)

#dependent variable 
y.dep <- 131

#independent variables
x.indep <- c(1:130)


system.time(gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = dev.h2o, ntrees = 100, max_depth = 10, nfolds = 5, learn_rate = 0.05, sample_rate = 0.8, col_sample_rate = 0.8, seed = 1122))
h2o.performance (gbm.model)
gbm.model@model$cross_validation_metrics_summary
h2o.mse(h2o.performance(gbm.model, xval = TRUE))

predict.gbm <- as.data.frame(h2o.predict(gbm.model, valid.h2o))
mean(abs( (exp(predict.gbm$predict)-200) - valid.data$loss))

predict.gbm1 <- as.data.frame(exp(h2o.predict(gbm.model, test.h2o)) - 200)
sub_gbm <- data.frame(id = IDs, loss = predict.gbm1$C1)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)

########################################### DL in H2O #####################################

splits <- h2o.splitFrame( data = dev.h2o, ratios = c(0.8), destination_frames = c("train.hex", "valid.hex"), seed = 1234 )
train <- splits[[1]]
valid <- splits[[2]]

dl_model <- h2o.deeplearning( training_frame = train, validation_frame = valid, x = x.indep, y = y.dep, overwrite_with_best_model = T, hidden = c(64,32), epochs = 10)
# h2o.saveModel(dl_model, path = getwd())
# h2o.loadModel(getwd())

summary(dl_model)
h2o.performance(dl_model)

predict.dl2 <- as.data.frame(exp(h2o.predict(dl_model, valid.h2o)) - 200)
mean(abs(predict.dl2$C1 - valid.data$loss))

#test data set prediction
predict.dl3 <- as.data.frame(exp(h2o.predict(dl_model, test.h2o)) - 200)
sub_dl <- data.frame(id = IDs, loss = predict.dl3$C1)
write.csv(sub_dl, file = "sub_dl.csv", row.names = F)

########################### Ensembling submissions ###########################################

xgb = read.csv("sub_xgb.csv")
xgbm4_fe = read.csv("sub_gbm.csv")
dl1 = read.csv("sub_dl.csv")
cor(xgb$loss, xgbm4_fe$loss, dl1$loss)
results = data.frame(id = dl1$id, xgb$loss, xgbm4_fe$loss, dl1$loss)
results = dplyr::mutate(results, ensemble = (xgb.loss + xgbm4_fe.loss + dl1.loss)/3)
correlations <- cor(results[,2:4])
corrplot(correlations, method="square", order="hclust")

sub_ensemble <- data.frame(id = IDs, loss = results$ensemble)
write.csv(sub_ensemble, file = "sub_ensemble1.csv", row.names = F)

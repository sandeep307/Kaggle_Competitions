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
setwd("../")

train.data = fread("AllState/train.csv")
train.data$train_flag = 1
test.data = fread("AllState/test.csv")
test.data$loss = 0
test.data$train_flag = 0

IDs = test.data$id
data = as.data.frame(rbind(train.data, test.data))
data$id = NULL

# EDA ---------------------------------------------------------------------
cat_var <- names(train.data)[which(sapply(train.data, is.character))]
num_var <- names(train.data)[which(sapply(train.data, is.numeric))]
num_var <- setdiff(num_var, c("id", "loss"))

#Boxplot function
plotBox <- function(data_in, i, lab) {
  data <- data.frame(x = data_in[[i]], y = lab)
  p <- ggplot(data = data, aes(x = x, y = y)) + geom_boxplot() + xlab(colnames(data_in)[i]) + theme_light() + ylab("log(loss)") + theme(axis.text.x = element_text(angle = 90, hjust = 1))
  return (p)
}

doPlots <- function(data_in, fun, ii, lab, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(data_in = data_in, i = i, lab = lab)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol = ncol))
}

plotScatter <- function(data_in, i, lab){
  data <- data.frame(x = data_in[[i]], y = lab)
  p <- ggplot(data = data, aes(x = x, y=y)) + geom_point(size=1, alpha=0.3)+ geom_smooth(method = lm) + xlab(paste0(colnames(data_in)[i], '\n', 'R-Squared: ', round(cor(data_in[[i]], lab, use = 'complete.obs'), 2)))+ ylab("log(loss)") + theme_light()
  return(suppressWarnings(p))
} 

plotDen <- function(data_in, i, lab){
  data <- data.frame(x=data_in[[i]], y=lab)
  p <- ggplot(data= data) + geom_density(aes(x = x), size = 1,alpha = 1.0) + xlab(paste0((colnames(data_in)[i]), '\n', 'Skewness: ',round(skewness(data_in[[i]], na.rm = TRUE), 2))) + theme_light() 
  return(p)
}


doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =1:12, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =13:24, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =25:36, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =37:48, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =49:60, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =61:72, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =73:84, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =85:96, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =97:108, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,cat_var, with = F], fun = plotBox, ii =109:116, lab=log(train.data$loss), ncol = 3)


#density plots
doPlots(train.data[,num_var, with = F], fun = plotDen, ii =1:6, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,num_var, with = F], fun = plotDen, ii =7:14, lab=log(train.data$loss), ncol = 3)

#Scatter plots
doPlots(train.data[,num_var, with = F], fun = plotScatter, ii =1:6, lab=log(train.data$loss), ncol = 3)
doPlots(train.data[,num_var, with = F], fun = plotScatter, ii =7:14, lab=log(train.data$loss), ncol = 3)


#Feature Engg.
train.data$mult_27 = train.data$cont2*train.data$cont7
train.data$mult_214 = train.data$cont2*train.data$cont14
train.data$mult_714 = train.data$cont7*train.data$cont14

train.data$div_27 = train.data$cont2/train.data$cont7
train.data$div_214 = train.data$cont2/train.data$cont14
train.data$div_714 = train.data$cont7/train.data$cont14

train.data$mult_2714 = train.data$cont2*train.data$cont7*train.data$cont14
doPlots(train.data[,134:139, with = F], fun = plotScatter, ii =1:6, lab=log(train.data$loss), ncol = 3)


correlations <- cor(train.data[,c(118:131,134:139), with = F])
corrplot(correlations, method="square", order="hclust")

ggplot(train.data) + geom_histogram(mapping=aes(x=log(loss)))

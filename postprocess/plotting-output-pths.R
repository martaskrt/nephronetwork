# lab desktop
setwd("C:/Users/Lauren/Desktop/DS Core/Projects/Urology/eval_nn/summary_files/")

# laptop
setwd("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/")

### PACKAGES

library("ggplot2")
library("lubridate")
library("pROC")
library("dismo")

### USER-DEFINED FUCTIONS

get_manu = function(full_id_vec){
  manu = unlist(lapply(strsplit(full_id_vec,split = "_"),function(x){x[length(x)]}))
  return(manu)
}

fix_gender = function(df){
  df_out = df
  df_out$date_of_current_us[!(df_out$gender %in% c("F","M"))] <- df_out$date_of_us1[!(df_out$gender %in% c("F","M"))]
  df_out$date_of_us1[!(df_out$gender %in% c("F","M"))] <- df_out$kidney_side[!(df_out$gender %in% c("F","M"))]
  df_out$kidney_side[!(df_out$gender %in% c("F","M"))] <- df_out$us_num[!(df_out$gender %in% c("F","M"))]
  df_out$us_num[!(df_out$gender %in% c("F","M"))] <- df_out$gender[!(df_out$gender %in% c("F","M"))]
  df_out$gender[!(df_out$gender %in% c("F","M"))] <- "F"

  return(df_out)
}

elapsed_months <- function(end_date, start_date) { ## from: https://stackoverflow.com/questions/1995933/number-of-months-between-two-dates/1996404
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}

get_cutpoint <- function(pos_class_vec,sensitivity = 0.95){
  sorted_vec = sort(pos_class_vec)
  cutpoint_thresh = quantile(sorted_vec,c(1-sensitivity))
  return(cutpoint_thresh)
}

get_pred_class <- function(pred_vals,threshold = 0.5){
  return(ifelse(pred_vals > threshold,yes = 1,no = 0))
}


### LOAD CLINICAL DATA
load("revised-us-classifier-Feb72019.RData") ## add path to this
  ## relevant dataframe = phn.raw ; want vcug1 (or all VCUG variables really)

### PROCESSING THE DATA

analysis_name = "unet_20190503_vanilla_CV_lr0.001_e35_bs256_c1_SGD_pi"

train = read.csv(paste0(analysis_name,"_train.csv"),header=TRUE,as.is=TRUE)
val = read.csv(paste0(analysis_name,"_val.csv"),header=TRUE,as.is=TRUE)
test = read.csv(paste0(analysis_name,"_test.csv"),header=TRUE,as.is=TRUE)
head(train)

data_triad = list("train" = train,"val" = val,"test" = test)
data_triad = lapply(data_triad,function(x){fix_gender(x)})
data_triad = lapply(data_triad,function(x){x$manu = get_manu(x[,"full_ID"]) ; return(x)})
str(data_triad)

full_dat = Reduce(rbind,data_triad)
str(full_dat)
length(unique(full_dat$full_ID))

## IF NO M/F IN GENDER, SHIFT RIGHT, ADD "F"

full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                 rep("val",nrow(data_triad[["val"]])),
                 rep("test",nrow(data_triad[["test"]])))
full_dat$Data_Split = factor(full_dat$set,levels = c("train","val","test"),labels = c("Training","Validation","Test"))
full_dat$Target.f = factor(full_dat$Target,levels = c(0,1),labels = c("No Surgery","Surgery"))
full_dat$date_of_us1.date = as.Date(full_dat$date_of_us1)
full_dat$date_of_current_us.date = as.Date(full_dat$date_of_current_us)
full_dat$date_of_current_us.date[full_dat$date_of_current_us.date > "2020-01-01"] = NA
full_dat$us_1_yr = year(full_dat$date_of_us1.date) ##  can extract year from lubridate function
full_dat$us_yr = year(full_dat$date_of_current_us.date) ##  can extract year from lubridate function
full_dat$us_yr[full_dat$us_yr > 2020] = NA
table(full_dat$kidney_side)
str(full_dat)
full_dat$age_at_us <- NA
full_dat$age_at_us = full_dat$age_at_baseline + elapsed_months(end_date = full_dat$date_of_current_us.date,start_date = full_dat$date_of_us1.date)
full_dat$us_num.f = factor(full_dat$us_num,levels = 1:10)
str(full_dat$us_num.f)

if(exists("phn.raw")){
  full_dat$vcug_yn = phn.raw$vcug1[match(full_dat$study_id,phn.raw$study_id)]
  full_dat$vcug_yn_0 = full_dat$vcug_yn
  full_dat$vcug_yn_0[is.na(full_dat$vcug_yn) == T] = 0
  full_dat$vcug_yn.f = factor(full_dat$vcug_yn,levels = c(0,1),labels = c("no","yes"))
  full_dat$vcug_yn_0.f = factor(full_dat$vcug_yn_0,levels = c(0,1),labels = c("no","yes"))
}


full_dat_un_test = full_dat[!duplicated(paste0(full_dat$full_ID[full_dat$Data_Split == "Test"],":",full_dat$Fold[full_dat$Data_Split == "Test"])),]
str(full_dat_un_test)
length(unique(full_dat$full_ID[full_dat$Data_Split == "Test"]))
length(full_dat$full_ID[full_dat$Data_Split == "Test"])
length(full_dat_un_test$full_ID[full_dat_un_test$Data_Split == "Test"])

full_dat[full_dat$study_id == 585,]

full_dat.sorted = full_dat[order(full_dat$date_of_current_us.date),]


ggplot(full_dat,aes(x = Target.f,y = Pred_val,size = age_at_baseline)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_1_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = age_at_us)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(vcug_yn.f~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = study_id)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)

### looking at sorted dataset 

ggplot(full_dat.sorted,aes(x = age_at_us,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)

ggplot(full_dat.sorted,aes(x = date_of_current_us.date,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)

ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = us_num)) + geom_point() + geom_jitter() + facet_grid(us_num~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)

###############################
###############################
#
#   TESTING A MODEL TO CORRECT OUTPUT FROM UNet
#
###############################
###############################

pos_class_vec = data_triad[["val"]]$Pred_val[data_triad[["val"]]$Target == 1]

thresh = get_cutpoint(pos_class_vec,sensitivity = 0.95)

pred_new_thresh = get_pred_class(data_triad[["test"]]$Pred_val,threshold = thresh)

table(pred_new_thresh,data_triad[["test"]]$Target)
table(pred_new_thresh,full_dat$vcug_yn.f[full_dat$Data_Split == "Test"])
na.omit(full_dat$study_id[full_dat$vcug_yn == 1 & full_dat$Data_Split == "Test"])
na.omit(full_dat$study_id[full_dat$vcug_yn == 0 & full_dat$Data_Split == "Test"])

###############################
###############################
#
#   TESTING A MODEL TO CORRECT OUTPUT FROM UNet -- unsuccessful 
#
###############################
###############################

pred_mod = glm(Target ~ Pred_val + kidney_side,data = data_triad[["val"]], family = binomial)
summary(pred_mod)

new_test_pred = predict(pred_mod,newdata = data_triad[["test"]],type = "response")

new_test_out = data.frame(pred = new_test_pred,
                          target = data_triad[["test"]]$Target)
new_test_out$target.f = factor(new_test_out$target,levels = c(0,1),labels = c("No Surgery","Surgery"))

auc(data_triad[["test"]]$Target,data_triad[["test"]]$Pred_val) ## 0.8346
ci.sp(data_triad[["test"]]$Target,data_triad[["test"]]$Pred_val)

e = evaluate(data_triad[["train"]]$Pred_val,data_triad[["train"]]$Target)
threshold(e)

auc(new_test_out$target,new_test_out$pred) ## 0.804

ggplot(new_test_out,aes(x = target.f,y = pred)) + geom_point() + geom_jitter()



# lab desktop
setwd("C:/Users/Lauren/Desktop/DS Core/Projects/Urology/eval_nn/summary_files/")

# laptop
setwd("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/")

### PACKAGES

library("ggplot2")
library("lubridate")

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

### PROCESSING THE DATA

analysis_name = "unet_20190503_vanilla_CV_lr0.001_e35_bs256_c1_SGD_test20_pi"

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
## IF NO M/F IN GENDER, SHIFT RIGHT, ADD "F"

full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                 rep("val",nrow(data_triad[["val"]])),
                 rep("test",nrow(data_triad[["test"]])))
full_dat$Data_Split = factor(full_dat$set,levels = c("train","val","test"))
full_dat$Target.f = factor(full_dat$Target,levels = c(0,1),labels = c("No Surgery","Surgery"))
full_dat$date_of_us1.date = as.Date(full_dat$date_of_us1)
full_dat$date_of_current_us.date = as.Date(full_dat$date_of_current_us)
full_dat$us_1_yr = year(full_dat$date_of_us1.date) ##  can extract year from lubridate function
full_dat$us_yr = year(full_dat$date_of_current_us.date) ##  can extract year from lubridate function
full_dat$us_yr[full_dat$us_yr > 2020] = NA
table(full_dat$kidney_side)
str(full_dat)
full_dat$age_at_us[!is.na(full_dat$us_yr)] = full_dat$age_at_baseline[!is.na(full_dat$us_yr)] + (full_dat$us_yr[!is.na(full_dat$us_yr)] - full_dat$us_1_yr[!is.na(full_dat$us_yr)])

ggplot(full_dat,aes(x = Target.f,y = Pred_val,size = age_at_baseline)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_1_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = age_at_us)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)

ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(.~study_id)



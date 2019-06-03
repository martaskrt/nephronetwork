# lab desktop
setwd("C:/Users/Lauren/Desktop/DS Core/Projects/Urology/eval_nn/summary_files/")

# laptop
setwd("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/")

### PACKAGES

library("ggplot2")
library("lubridate")
library("pROC")
library("PRROC")
#library("dismo")

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

simple_roc <- function(labels, scores){
  labels <- labels[order(scores, decreasing=TRUE)]
  data.frame(TPR=cumsum(labels)/sum(labels), FPR=cumsum(!labels)/sum(!labels), labels)
}

### LOAD CLINICAL DATA
  ## lab desktop

  ## laptop
rdata_path = "C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/"

### UPDATE THIS FILE
obstr.data = read.csv(paste0(rdata_path,"PHN-ObstructiveEtiologie_Labels.csv"),
                      header=TRUE,as.is=TRUE)
reflux.data = read.csv(paste0(rdata_path,"PHN-ObstructiveEtiologie_Labels.csv"),
                      header=TRUE,as.is=TRUE)

phn.raw = data.frame(rbind(obstr.data,reflux.data))
  ## relevant dataframe = phn.raw ; want vcug1 (or all VCUG variables really)
phn.raw$kidney_side = ifelse(test = (phn.raw$Laterality == "Bilateral"),
                             yes = phn.raw$If.bilateral..which.is.the.most.severe.kidney.,
                             no = phn.raw$Laterality)
str(phn.raw$kidney_side)


### PROCESSING THE DATA

# analysis_name = "unet_20190517_vanilla_CV_crop0_OR_plusRef_BN"
# analysis_name = "auc0.91unet_run"
analysis_name = "unet_vanilla_both_20190527"
cv = FALSE

train = read.csv(paste0(analysis_name,"_train.csv"),header=TRUE,as.is=TRUE)
if(cv){
  val = read.csv(paste0(analysis_name,"_val.csv"),header=TRUE,as.is=TRUE)
}
test = read.csv(paste0(analysis_name,"_test.csv"),header=TRUE,as.is=TRUE)
head(train)
head(test)

if(cv){
  data_triad = list("train" = train,"val" = val,"test" = test)
} else{
  data_triad = list("train" = train,"test" = test)
}

data_triad = lapply(data_triad,function(x){fix_gender(x)})
data_triad = lapply(data_triad,function(x){x$manu = get_manu(x[,"full_ID"]) ; return(x)})
str(data_triad)

full_dat = Reduce(rbind,data_triad)
if(cv){
  full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                   rep("val",nrow(data_triad[["val"]])),
                   rep("test",nrow(data_triad[["test"]])))
  full_dat$Data_Split = factor(full_dat$set,levels = c("train","val","test"),labels = c("Training","Validation","Test"))
} else{
  full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                   rep("test",nrow(data_triad[["test"]])))
  full_dat$Data_Split = factor(full_dat$set,levels = c("train","test"),labels = c("Training","Test"))
}
str(full_dat)
full_dat$us_num = as.numeric(full_dat$us_num)
length(unique(full_dat$full_ID))

length(unique(full_dat$study_id))
length(unique(data_triad$train$study_id))
length(unique(data_triad$test$study_id))

## MAKE DATASET OF HYDRO KIDNEYS ONLY: 
  ## INCLUDE SFU GRADE, VCUG Y/N, 
str(full_dat)
str(phn.raw)
names(phn.raw)
phn.raw.nodup = phn.raw[!duplicated(phn.raw$Study.ID),]
str(phn.raw.nodup)
new.cols = c("SFU_grade","APD","ERP","Ureter.Dilation",
             "Etiology","UTI","renal_scan1","renal_scan2",
             "renal_scan3","surgery_type","surgery_type2")
hydro_only_full_dat = data.frame(matrix(nrow=0,ncol=ncol(full_dat)+length(new.cols)))
names(hydro_only_full_dat) <- c(names(full_dat),new.cols)
str(hydro_only_full_dat)
names(hydro_only_full_dat)

## Renal scan  = nuclear scan 
  ##   -- can get multiple nuclear scans

row = 1
k = 1
for(row in 1:nrow(full_dat)){
  if(paste0(full_dat$study_id[row],":",full_dat$kidney_side[row]) %in% paste0(phn.raw$Study.ID,":",phn.raw$kidney_side)){
    X = k
    Pred_val = full_dat$Pred_val[row]
    Target = full_dat$Target[row]
    age_at_baseline = full_dat$age_at_baseline[row]
    date_of_current_us = full_dat$date_of_current_us[row]
    date_of_us1 = full_dat$date_of_us1[row]
    full_ID = full_dat$full_ID[row]
    gender = full_dat$gender[row]
    kidney_side = full_dat$kidney_side[row]
    study_id = full_dat$study_id[row]
    cat("study id:\n")
    cat(paste0(study_id,"\n"))
    us_num = full_dat$us_num[row]
    cat("us number: \n")
    cat(paste0(us_num,"\n"))
    manu = full_dat$manu[row]
    set = full_dat$set[row]
    Data_Split = full_dat$Data_Split[row]
    if(us_num == 1){
      SFU_grade = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"SFU.Classification"]
      APD = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"APD"]
      ERP = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"ERP.diamater"]
    } else if(us_num == 2){
      SFU_grade = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"SFU.Grade"]
      APD = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("APD.",us_num-1)]
      ERP = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("ERP.diamater.",us_num-1)]
    } else{
      SFU_grade = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("SFU.Grade.",us_num-2)]
      APD = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("APD.",us_num-1)]
      ERP = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("ERP.diamater.",us_num-1)]
    }
    # cat("SFU grade:\n")
    # cat(paste0(SFU_grade,"\n"))
    Ureter.Dilation = ifelse(is.null(phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("Ureter.Dilation.",us_num)]),
                             yes = NA,
                             no = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,paste0("Ureter.Dilation.",us_num)])
    Etiology = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Etiology"]
    UTI = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"UTI"]
    renal_scan1 = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Renal.Scan.1"]
    renal_scan2 = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Renal.Scan.2"]
    renal_scan3 = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Renal.Scan.3"]
    surgery_type = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Type"]
    surgery_type2 = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Type.1"]
    
    in_row = c(X,Pred_val,Target,age_at_baseline,date_of_current_us,
               date_of_us1,full_ID,gender,kidney_side,study_id,
               us_num,manu,set, Data_Split,SFU_grade,APD,ERP,
               Ureter.Dilation,Etiology,UTI,renal_scan1,renal_scan2,
               renal_scan3,surgery_type,surgery_type2)
    hydro_only_full_dat[k,] <- in_row
    k = k+1
  }
}

str(hydro_only_full_dat)
# write.table(unique(full_dat$study_id),"all-samples-May232019.txt",
#             quote=FALSE,row.names=FALSE,col.names=FALSE)
# write.table(unique(data_triad$train$study_id),"train-samples-May232019.txt",
#             quote=FALSE,row.names=FALSE,col.names=FALSE)
# write.table(unique(data_triad$test$study_id),"test-samples-May232019.txt",
#             quote=FALSE,row.names=FALSE,col.names=FALSE)


table(data_triad$test$Target[!duplicated(data_triad$test$study_id)])
table(data_triad$train$Target[!duplicated(data_triad$train$study_id)])/sum(table(data_triad$train$Target[!duplicated(data_triad$train$study_id)]))
table(data_triad$test$Target[!duplicated(data_triad$test$study_id)])/sum(table(data_triad$test$Target[!duplicated(data_triad$test$study_id)]))

## IF NO M/F IN GENDER, SHIFT RIGHT, ADD "F"

  ## REVISING VARIABLES FOR FULL DATA
if(cv){
  full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                   rep("val",nrow(data_triad[["val"]])),
                   rep("test",nrow(data_triad[["test"]])))
  full_dat$Data_Split = factor(full_dat$set,levels = c("train","val","test"),labels = c("Training","Validation","Test"))
} else{
  full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                   rep("test",nrow(data_triad[["test"]])))
  full_dat$Data_Split = factor(full_dat$set,levels = c("train","test"),labels = c("Training","Test"))
}
full_dat$Target.f = factor(full_dat$Target,levels = c(0,1),labels = c("No Surgery","Surgery"))
full_dat$date_of_us1.date = as.Date(full_dat$date_of_us1)
full_dat$date_of_current_us.date = as.Date(full_dat$date_of_current_us)
full_dat$date_of_current_us.date[full_dat$date_of_current_us.date > "2020-01-01"] = NA
full_dat$us_1_yr = year(full_dat$date_of_us1.date) ##  can extract year from lubridate function
full_dat$us_yr = year(full_dat$date_of_current_us.date) ##  can extract year from lubridate function
full_dat$us_yr[full_dat$us_yr > 2020] = NA
table(full_dat$kidney_side)
str(full_dat)

full_dat$age_at_us = full_dat$age_at_baseline + elapsed_months(end_date = full_dat$date_of_current_us.date,start_date = full_dat$date_of_us1.date)
full_dat$us_num.f = factor(full_dat$us_num,levels = 1:10)
str(full_dat$us_num.f)
# full_dat_un_test = full_dat[!duplicated(paste0(full_dat$full_ID[full_dat$Data_Split == "Test"],":",full_dat$Fold[full_dat$Data_Split == "Test"])),]
  ## END REVISING VARIABLES FOR FULL DATA

  ## REVISING VARIABLES FOR HYDRO ONLY DATA 
str(hydro_only_full_dat)
hydro_only_full_dat$Pred_val <- as.numeric(hydro_only_full_dat$Pred_val)
hydro_only_full_dat$age_at_baseline <- as.numeric(hydro_only_full_dat$age_at_baseline)
hydro_only_full_dat$age_at_us <- as.numeric(hydro_only_full_dat$age_at_us)
hydro_only_full_dat$SFU_grade <- factor(hydro_only_full_dat$SFU_grade,levels = as.character(0:4))
hydro_only_full_dat$APD <- as.numeric(hydro_only_full_dat$APD)
hydro_only_full_dat$ERP <- as.numeric(hydro_only_full_dat$ERP)
if(cv){
  hydro_only_full_dat$Data_Split = factor(hydro_only_full_dat$Data_Split,
                                          levels = 1:3,
                                          labels = c("Training","Validation","Test"))  
} else{
  hydro_only_full_dat$Data_Split = factor(hydro_only_full_dat$Data_Split,
                                          levels = 1:2,
                                          labels = c("Training","Test"))  
}

hydro_only_full_dat$Target.f = factor(hydro_only_full_dat$Target,levels = c(0,1),labels = c("No Surgery","Surgery"))
hydro_only_full_dat$date_of_us1.date = as.Date(hydro_only_full_dat$date_of_us1)
hydro_only_full_dat$date_of_current_us.date = as.Date(hydro_only_full_dat$date_of_current_us)
hydro_only_full_dat$date_of_current_us.date[hydro_only_full_dat$date_of_current_us.date > "2020-01-01"] = NA
hydro_only_full_dat$us_1_yr = year(hydro_only_full_dat$date_of_us1.date) ##  can extract year from lubridate function
hydro_only_full_dat$us_yr = year(hydro_only_full_dat$date_of_current_us.date) ##  can extract year from lubridate function
hydro_only_full_dat$us_yr[hydro_only_full_dat$us_yr > 2020] = NA
table(hydro_only_full_dat$kidney_side)
str(hydro_only_full_dat)

hydro_only_full_dat$age_at_us = as.numeric(hydro_only_full_dat$age_at_baseline) + elapsed_months(end_date = hydro_only_full_dat$date_of_current_us.date,start_date = hydro_only_full_dat$date_of_us1.date)
hydro_only_full_dat$us_num.f = factor(hydro_only_full_dat$us_num,levels = 1:10)
str(hydro_only_full_dat$us_num.f)
  ## END REVISING VARIABLES FOR HYDRO ONLY DATA 

### GETTING MIS-DIAGNOSED PATIENTS
hydro_only_full_dat[hydro_only_full_dat$Target == 1 & 
                      hydro_only_full_dat$Data_Split == "Test" & 
                      hydro_only_full_dat$surgery_type == "Circumcision",
                    c("study_id","us_num","kidney_side","surgery_type")]

hydro_only_full_dat[hydro_only_full_dat$Target == 1 & 
                      hydro_only_full_dat$Data_Split == "Training" & 
                      hydro_only_full_dat$surgery_type == "Circumcision",
                    c("study_id","us_num","kidney_side","surgery_type","surgery_type2")]


hydro_only_full_dat[hydro_only_full_dat$Target == 0 & hydro_only_full_dat$Data_Split == "Test" & hydro_only_full_dat$Pred_val > 0.6,c("study_id","us_num","kidney_side")]


  ## GRAPHS ON HYDRO ONLY DATASET
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = SFU_grade)) + geom_point() + geom_jitter() + facet_grid(SFU_grade~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = Ureter.Dilation)) + geom_point() + geom_jitter() + facet_grid(Ureter.Dilation~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = APD)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,col = ERP)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,fill = Etiology)) + geom_violin() + facet_grid(Etiology~Data_Split)
ggplot(hydro_only_full_dat,aes(x = Target.f,y = Pred_val,fill = UTI)) + geom_violin() + facet_grid(UTI~Data_Split)

  ## GRAPHS ON FULL DATASET
## unsorted data graphs
ggplot(full_dat,aes(x = Target.f,y = Pred_val,size = age_at_baseline)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_1_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_yr)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = age_at_us)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = study_id)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split)
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)

  ### looking at sorted dataset 
full_dat.sorted = full_dat[order(full_dat$date_of_current_us.date),]

ggplot(full_dat.sorted,aes(x = age_at_us,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)
ggplot(full_dat.sorted,aes(x = date_of_current_us.date,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,fill = manu)) + geom_violin() + facet_grid(manu~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = us_num)) + geom_point() + geom_jitter() + facet_grid(us_num~Data_Split)
ggplot(full_dat.sorted,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)


###############################
###############################
#
#   Investigating accuracy by ultrasound number
#
###############################
###############################

  ## Plot of data by ultrasound number 
ggplot(full_dat,aes(x = Target.f,y = Pred_val,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split)
us_num_viol = ggplot(full_dat,aes(x = Target.f,y = Pred_val, fill = us_num.f)) + geom_violin() + facet_grid(us_num.f~Data_Split) + 
  labs(fill = "US Number") + xlab("Target") + ylab("Prediction") + 
  theme(strip.text.x = element_text(size = 15),
        axis.title = element_text(size = 15),
        axis.text.x = element_text(size = 13))
us_num_viol
ggplot_cols = unique(ggplot_build(us_num_viol)$data[[1]]$fill)


auc(data_triad[["test"]]$Target[data_triad[["test"]]$us_num == 1],data_triad[["test"]]$Pred_val[data_triad[["test"]]$us_num == 1]) ## 0.8993
auc(data_triad[["test"]]$Target[data_triad[["test"]]$us_num == 2],data_triad[["test"]]$Pred_val[data_triad[["test"]]$us_num == 2]) ## 0.9256
auc(data_triad[["test"]]$Target[data_triad[["test"]]$us_num == 3],data_triad[["test"]]$Pred_val[data_triad[["test"]]$us_num == 3]) ## 0.9677
auc(data_triad[["test"]]$Target[data_triad[["test"]]$us_num == 4],data_triad[["test"]]$Pred_val[data_triad[["test"]]$us_num == 4]) ## 0.9677

  ##### AUC CURVE FOR EACH US NUMBER
    ## TRAIN
us_auc_df = data.frame(matrix(nrow=0,ncol=4))
us_num_levels = 1:7
for(us_num in us_num_levels){
  df = simple_roc(data_triad[["train"]]$Target[data_triad[["train"]]$us_num == us_num],
                  data_triad[["train"]]$Pred_val[data_triad[["train"]]$us_num == us_num])
  df$us_num = us_num
  
  us_auc_df = rbind(us_auc_df,df)
}
us_auc_df$us_num = factor(us_auc_df$us_num,levels = us_num_levels)

ggplot(us_auc_df,aes(x = FPR,y = TPR,col = us_num)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("1" = ggplot_cols[1],
                                "2" = ggplot_cols[2],
                                "3" = ggplot_cols[3],
                                "4" = ggplot_cols[4],
                                "5" = ggplot_cols[5],
                                "6" = ggplot_cols[6],
                                "7" = ggplot_cols[7]),
                     name = "US Number") + 
  theme(axis.title = element_text(size = 15))

    ##  TEST
us_auc_df = data.frame(matrix(nrow=0,ncol=4))
us_num_levels = 1:3
for(us_num in us_num_levels){
  df = simple_roc(data_triad[["test"]]$Target[data_triad[["test"]]$us_num == us_num],
                  data_triad[["test"]]$Pred_val[data_triad[["test"]]$us_num == us_num])
  df$us_num = us_num

  us_auc_df = rbind(us_auc_df,df)
}
us_auc_df$us_num = factor(us_auc_df$us_num,levels = us_num_levels)

ggplot(us_auc_df,aes(x = FPR,y = TPR,col = us_num)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("1" = ggplot_cols[1],
                                "2" = ggplot_cols[2],
                                "3" = ggplot_cols[3]),
                     name = "US Number") + 
  theme(axis.title = element_text(size = 15))


 ## INVESTIGATING SPECIFIC CASES


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



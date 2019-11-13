
# lab desktop
setwd("C:/Users/Lauren/Desktop/DS Core/Projects/Urology/eval_nn/summary_files/")

# laptop
setwd("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/")

### PACKAGES
library("reshape")
library("ggplot2")
# library("lubridate")
library("pROC")
library("PRROC")
#library("dismo")

### USER-DEFINED FUCTIONS

load_data = function(analysis_name,cv=FALSE){
  train = read.csv(paste0(analysis_name,"_train.csv"),header=TRUE,as.is=TRUE)
  if(cv){
    val = read.csv(paste0(analysis_name,"_val.csv"),header=TRUE,as.is=TRUE)
  }
  test = read.csv(paste0(analysis_name,"_test.csv"),header=TRUE,as.is=TRUE)
  
  if(cv){
    data_triad = list("train" = train,"val" = val,"test" = test)
  } else{
    data_triad = list("train" = train,"test" = test)
  }
  
  data_triad = lapply(data_triad,function(x){fix_gender(x)})
  data_triad = lapply(data_triad,function(x){x$manu = get_manu(x[,"full_ID"]) ; return(x)})
  
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
  
  full_dat$us_num = as.numeric(full_dat$us_num)
  
  out_list = list("data_triad" = data_triad,
                  "full_dat" = full_dat)
  
  return(out_list)
}

make_hydro_df = function(full_df,phn_df,new_cols = c("SFU_grade","APD","ERP","Ureter.Dilation",
                                                     "Etiology","UTI","renal_scan1","renal_scan2",
                                                     "renal_scan3","surgery_type","surgery_type2",
                                                     "anomalies")){
  phn.raw.nodup = phn_df[!duplicated(phn_df$Study.ID),]
  phn.raw.nodup = phn.raw.nodup[!is.na(phn.raw$Study.ID),]
  hydro_only_full_dat = data.frame(matrix(nrow=0,ncol=ncol(full_df)+length(new_cols)))
  names(hydro_only_full_dat) <- c(names(full_df),new_cols)
  
  
  ## Renal scan  = nuclear scan 
  ##   -- can get multiple nuclear scans
  
  row = 1
  k = 1
  for(row in 1:nrow(full_df)){
    if(paste0(full_df$study_id[row],":",full_df$kidney_side[row]) %in% paste0(phn.raw.nodup$Study.ID,":",phn.raw.nodup$kidney_side)){
      X = k
      Pred_val = full_df$Pred_val[row]
      Target = full_df$Target[row]
      age_at_baseline = full_df$age_at_baseline[row]
      date_of_current_us = full_df$date_of_current_us[row]
      date_of_us1 = full_df$date_of_us1[row]
      full_ID = full_df$full_ID[row]
      gender = full_df$gender[row]
      kidney_side = full_df$kidney_side[row]
      study_id = full_df$study_id[row]
      cat("study id:\n")
      cat(paste0(study_id,"\n"))
      us_num = full_df$us_num[row]
      cat("us number: \n")
      cat(paste0(us_num,"\n"))
      manu = full_df$manu[row]
      set = full_df$set[row]
      Data_Split = full_df$Data_Split[row]
      anomalies = phn.raw.nodup[phn.raw.nodup$Study.ID == study_id,"Anomalies"]
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
      cat("SFU grade:\n")
      cat(paste0(SFU_grade,"\n"))
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
      
      in_row = c(SFU_grade,APD,ERP,
                 Ureter.Dilation,Etiology,UTI,renal_scan1,renal_scan2,
                 renal_scan3,surgery_type,surgery_type2,anomalies)


      hydro_only_full_dat[k,] <- c(full_df[row,],in_row)
      k = k+1
    }
  }
  
  return(hydro_only_full_dat)  
}


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

revise_full_data = function(full_df,data_triad,cv,scale_model=NULL){
  data_triad_sub = lapply(data_triad,function(x){x[paste0(x[,"study_id"],":",x[,"kidney_side"]) %in% paste0(full_df$study_id,":",full_df$kidney_side),]})
  if(cv){
    full_df$set = c(rep("train",nrow(data_triad_sub[["train"]])),
                    rep("val",nrow(data_triad_sub[["val"]])),
                    rep("test",nrow(data_triad_sub[["test"]])))
    full_df$Data_Split = factor(full_df$set,
                                levels = c("train","val","test"),
                                labels = c("Training","Validation","Test"))
  } else{
    full_df$set = c(rep("train",nrow(data_triad_sub[["train"]])),
                    rep("test",nrow(data_triad_sub[["test"]])))
    full_df$Data_Split = factor(full_df$set,
                                levels = c("train","test"),
                                labels = c("Training","Test"))
  }
  full_df$Target.f = factor(full_df$Target,
                            levels = c(0,1),
                            labels = c("No Surgery","Surgery"))
  full_df$date_of_us1.date = as.Date(full_df$date_of_us1)
  full_df$date_of_current_us.date = as.Date(full_df$date_of_current_us)
  full_df$date_of_current_us.date[full_df$date_of_current_us.date > "2020-01-01"] = NA
  full_df$us_1_yr = year(full_df$date_of_us1.date) ##  can extract year from lubridate function
  full_df$us_yr = year(full_df$date_of_current_us.date) ##  can extract year from lubridate function
  full_df$us_yr[full_df$us_yr > 2020] = NA
  
  full_df$age_at_us = full_df$age_at_baseline + elapsed_months(end_date = full_df$date_of_current_us.date,start_date = full_df$date_of_us1.date)
  full_df$us_num.f = factor(full_df$us_num,levels = 1:10)
  
  if(!is.null(scale_model)){
    full_df$Scaled_Pred = predict(object = scale_model,newdata=full_df,type="response")
  }
  
  return(full_df)  
}

### LOAD CLINICAL DATA

### PHN CLINICAL FILE 
phn.raw = read.csv("20190524_combinedObstRef.csv", #sep = "\t",
                      header=TRUE,as.is=TRUE)

  ## relevant dataframe = phn.raw ; want vcug1 (or all VCUG variables really)
phn.raw$kidney_side = ifelse(test = (phn.raw$Laterality == "Bilateral"),
                             yes = phn.raw$If.bilateral..which.is.the.most.severe.kidney.,
                             no = phn.raw$Laterality)
str(phn.raw$kidney_side)
# phn.raw$Study.ID = phn.raw$ï..Study.ID

### PROCESSING THE CV DATA -- creating calibration map 
# my_analysis_name = "20190612_vanilla_siamese"
# my_cv = TRUE
# 
# out_cv = load_data(analysis_name = my_analysis_name,cv = my_cv)
# 
# cv_data_triad = out_cv$data_triad
# cv_full_dat = out_cv$full_dat
# cv_hydro_only = make_hydro_df(cv_full_dat,phn.raw)
# cv_hydro_revised_df = revise_full_data(full_df = cv_full_dat,data_triad = cv_data_triad,cv = my_cv)
# 
# 
# head(cv_full_dat)
# head(cv_hydro_only)
# str(cv_hydro_revised_df)
# 
# 
# platt_model = glm(Target ~ Pred_val,
#                   data = cv_hydro_revised_df[cv_hydro_revised_df$Data_Split == "Validation",],
#                   family = binomial(link=logit))
# scaled_target = predict(object = platt_model,type = "response")
# summary(scaled_target)


### PROCESSING THE FULL DATA
  ## full data
my_analysis_name = "./p-val-calcs/prehdict_20190802_vanilla_siamese_dim256_c1"
  ## hydro only
my_analysis_name = "20190618_vanilla_siamese_hydroonly_full"
full_cv = FALSE

out_full = load_data(analysis_name = my_analysis_name,cv = full_cv)

full_data_triad = out_full$data_triad
full_dat = out_full$full_dat

full_1 = revise_full_data(full_df = full_dat,
                                         data_triad = full_data_triad,
                                         cv = full_cv)

platt_model = glm(Target ~ Pred_val,
                   data = full_1[full_1$Data_Split == "Training",],
                   family = binomial(link=logit))

full_scaled2 = revise_full_data(full_df = full_dat,
                               data_triad = full_data_triad,
                               cv = full_cv,
                               scale_model = platt_model)
write.csv(full_scaled2,file=paste0(my_analysis_name,"_with_scaled_preds.csv"),
          quote=FALSE,row.names=FALSE)

full_hydro_only = make_hydro_df(full_df = full_scaled2,phn_df = phn.raw)
full_hydro_only_rev = revise_full_data(full_df = full_hydro_only,
                                       data_triad = full_data_triad,
                                       cv = full_cv)
hydro_only_platt = glm(Target ~ Pred_val,
                       data = full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training",],
                       family = binomial(link=logit))

full_scaled3 = revise_full_data(full_df = full_dat,
                                data_triad = full_data_triad,
                                cv = full_cv,
                                scale_model = hydro_only_platt)
full_hydro_only3 = make_hydro_df(full_df = full_scaled3,phn_df = phn.raw)
full_hydro_only_rev3 = revise_full_data(full_df = full_hydro_only3,
                                       data_triad = full_data_triad,
                                       cv = full_cv)

str(full_hydro_only)
# str(full_hydro_revised_df)
str(full_hydro_only_rev)

table(full_dat$Target[full_dat$Data_Split == "Training"])/sum(table(full_dat$Target[full_dat$Data_Split == "Training"]))
table(full_dat$Target[full_dat$Data_Split == "Test"])/sum(table(full_dat$Target[full_dat$Data_Split == "Test"]))

# auc(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"], 
#     full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test"])

    #######################
    ## HYDRO ONLY SUBSET SFU GRADE 3+4
    #######################
head(full_hydro_only_rev)
new_d = full_hydro_only_rev
new_d$ptn_id = unlist(lapply(strsplit(x = new_d$full_ID,split = "_"),function(x){as.numeric(x[1])}))
head(new_d)

new_d$my_ptn_id = paste0(new_d$ptn_id,"_",new_d$us_num,"_",new_d$kidney_side)
new_d$my_ptn_id

my_file_list = paste0(new_d$my_ptn_id,".jpg")

setwd("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/image-examples/sag-test-jpgs/")
file.copy(my_file_list,"C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/image-examples/high_ac_sag_jpgs/")

    #######################
    ## OVERALL BARPLOT OF DATA FOR THE MANUSCRIPT
    #######################
str(full_dat)
str(full_1)

barplot(table(full_1$us_1_yr),
        las=2,col = c("dodgerblue3","deeppink2")[full_1$Data_Split])

train.cts = melt(table(full_1$us_1_yr[full_1$Data_Split == "Training"]))
train.cts$Data_Split = "Training"
test.cts = melt(table(full_1$us_1_yr[full_1$Data_Split == "Test"]))
test.cts$Data_Split = "Test"

full.cts = rbind(train.cts,test.cts)
head(full.cts)
names(full.cts)[c(1,2)] = c("year","count")
ggplot(full.cts,aes(x = year,y = count,fill = Data_Split)) + geom_bar(stat = "identity")

ggplot(full_1,aes(x = date_of_us1.date,fill = Data_Split)) + 
  geom_bar(binwidth = 50) + xlab("Date of First Ultrasound") + 
  theme_bw()

    #######################
    ## HYDRO DATA GRAPHS
    #######################

  ## GRAPHS ON HYDRO ONLY DATASET
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = Data_Split)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1))
full_hydro_only_rev[full_hydro_only_rev$Target == 1 &
                      full_hydro_only_rev$Data_Split == "Test" &
                      full_hydro_only_rev$Scaled_Pred > 0.5,
                    c("study_id","kidney_side","us_num")]
# full_hydro_only_rev[full_hydro_only_rev$Target == 1 & 
#                       full_hydro_only_rev$Data_Split == "Test" & 
#                       full_hydro_only_rev$Scaled_Pred < 0.01,
#                     c("study_id","kidney_side","us_num")]
full_hydro_only_rev[full_hydro_only_rev$Target == 0 &
                      full_hydro_only_rev$Data_Split == "Test" &
                      full_hydro_only_rev$Scaled_Pred > 0.12,
                    c("study_id","kidney_side","us_num")]



#ggplot(full_hydro_only_rev,aes(x = Target.f,y = Pred_val,col = Data_Split)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1))

ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = Data_Split)) + geom_point() + geom_boxplot() + facet_grid(.~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = Data_Split)) + geom_point() + geom_violin() + facet_grid(.~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = Data_Split)) + geom_violin() + facet_grid(.~Data_Split) + ylim(c(0,1))

# dim(full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Test" &
#                           full_hydro_only_rev$Target == 1 & 
#                           full_hydro_only_rev$Scaled_Pred < 0.125,])
# dim(full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Test" &
#                           full_hydro_only_rev$Target == 1 & 
#                           full_hydro_only_rev$Scaled_Pred > 0.125,])
# 
# rsk <- full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test"]
# y <- full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"]
# num <- 0
# den <- 0
# for (ii in which(y == 1)) {
#   num <- num +  sum(rsk[ii] > rsk[y == 0])
#   den <- den + sum(y==0)
#   
# }
# num / den

ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = surgery_type)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1)) + facet_grid(surgery_type~Data_Split)

# full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training" &
#                       full_hydro_only_rev$Target == 1 & 
#                       full_hydro_only_rev$surgery_type == "Deflux Injection",]
# full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training" &
#                       full_hydro_only_rev$Target == 0 & 
#                       full_hydro_only_rev$surgery_type == "Ureterocystostomy",]
# full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training" &
#                       full_hydro_only_rev$Target == 0 & 
#                       full_hydro_only_rev$surgery_type == "Ureteral Reimplantion",]
# full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training" &
#                       full_hydro_only_rev$Target == 0 & 
#                       full_hydro_only_rev$surgery_type == "Ligation/clipping of ureter",]


ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split) + ylim(c(0,1))

full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Test" &
                      full_hydro_only_rev$Target == 1 &
                      full_hydro_only_rev$us_num == 3 &
                      full_hydro_only_rev$Scaled_Pred < 0.5,c("study_id","kidney_side","Scaled_Pred")]

ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = anomalies)) + geom_point() + geom_jitter() + facet_grid(anomalies~Data_Split) + ylim(c(0,1))

ggplot(full_hydro_only_rev[!is.na(full_hydro_only_rev$SFU_grade),],aes(x = Target.f,y = Scaled_Pred,col = SFU_grade)) + geom_point() + geom_jitter() + facet_grid(SFU_grade~Data_Split) + ylim(c(0,1))
    ### Who are we misclassifying in the grade 2 and grade 3 surgery? :( )
auc(full_hydro_only_rev$Target[full_hydro_only_rev$SFU_grade == 4],
    full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$SFU_grade == 4])

ggplot(full_hydro_only_rev[!is.na(full_hydro_only_rev$Ureter.Dilation),],aes(x = Target.f,y = Scaled_Pred,col = Ureter.Dilation)) + geom_point() + geom_jitter() + facet_grid(Ureter.Dilation~Data_Split) + ylim(c(0,1))
full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Test" & 
                      full_hydro_only_rev$Target == 1 & 
                      full_hydro_only_rev$Ureter.Dilation == "Yes",]


ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = APD)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = ERP)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,fill = Etiology)) + geom_violin() + facet_grid(Etiology~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = Etiology)) + geom_point() + geom_jitter() + facet_grid(Etiology~Data_Split) + ylim(c(0,1))
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,fill = UTI)) + geom_violin() + facet_grid(UTI~Data_Split) + ylim(c(0,1))

ggplot(full_hydro_only_rev,aes(x = Target.f,y = Scaled_Pred,col = UTI)) + geom_point() + geom_jitter() + facet_grid(UTI~Data_Split) + ylim(c(0,1))

ggplot(full_hydro_only_rev[!is.na(full_hydro_only_rev$Ureter.Dilation) & full_hydro_only_rev$Ureter.Dilation != "",],aes(x = Target.f,y = Scaled_Pred,col = Ureter.Dilation)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split) + ylim(c(0,1))


## SORTED DATA GRAPHS
hydro.sorted = full_hydro_only_rev[order(full_hydro_only_rev$date_of_current_us.date),]

ggplot(hydro.sorted,aes(x = age_at_us,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)

##############################
### GETTING HYDRO THRESHOLDS 
##############################

      ### With ensembled Scaled Pred: 
  ## prehdict siam
ens.scaled.pred = read.csv("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/p-val-calcs/predict_mod_scaled-test-201909.csv",
                           header = TRUE,as.is = TRUE)
ens_file = FALSE
  ## densenet st ensemble
# ens.scaled.pred = read.csv("C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/post-hoc-nn-eval/p-val-calcs/test_dense_st_ens-test-20190905.csv",
#                            header = TRUE,as.is = TRUE)
# ens_file = TRUE
str(full_hydro_only_rev)
str(ens.scaled.pred)

if(!ens_file){
  ens.scaled.pred$study_us_num = paste0(ens.scaled.pred$study_id,"_",
                                        ens.scaled.pred$us_num,"_",
                                        ens.scaled.pred$kidney_side)
}
full_hydro_only_rev$study_us_num = paste0(full_hydro_only_rev$study_id,"_",
                                          full_hydro_only_rev$us_num,"_",
                                          full_hydro_only_rev$kidney_side)
full_hydro_only_rev$Scaled_Pred = ens.scaled.pred$Scaled_Pred[match(full_hydro_only_rev$study_us_num,
                                                                    ens.scaled.pred$study_us_num)]
get_auc(full_hydro_only_rev)
get_auc(ens.scaled.pred)

get_auprc(full_hydro_only_rev)
get_auprc(ens.scaled.pred)

# pos_class_vec = data_triad[["val"]]$Pred_val[data_triad[["val"]]$Target == 1]
pos_class_vec = full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Training" & full_hydro_only_rev$Target == 1]
pos_class_vec = full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$Target == 1]

thresh = get_cutpoint(pos_class_vec,sensitivity = 0.95)

# pred_new_thresh = get_pred_class(data_triad[["test"]]$Pred_val,threshold = thresh)
pred_new_thresh = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test"],threshold = thresh)

# table(pred_new_thresh,data_triad[["test"]]$Target)
table(pred_new_thresh,
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"])

table(pred_new_thresh,
      full_hydro_only_rev$us_num[full_hydro_only_rev$Data_Split == "Test"],
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"])
## renal scan breakdown
table(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num == 1],
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num == 1])

unq.inds.tn = unique(full_hydro_only_rev$study_id[full_hydro_only_rev$Scaled_Pred < thresh & 
                                    full_hydro_only_rev$Target == 0 & 
                                    full_hydro_only_rev$Data_Split == "Test" & 
                                      full_hydro_only_rev$us_num %in% c(1,2)])

renal_scan_unq.inds.tn = phn.raw[phn.raw$Study.ID %in% unq.inds.tn,c("Study.ID","Renal.Scan.1")]
renal_scan_unq.inds.tn = renal_scan_unq.inds.tn[!duplicated(renal_scan_unq.inds.tn),]
table(renal_scan_unq.inds.tn$Renal.Scan.1)/sum(table(renal_scan_unq.inds.tn$Renal.Scan.1))

table(pred_new_thresh,
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"],
      full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test"])
table(pred_new_thresh,
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"],
      full_hydro_only_rev$renal_scan2[full_hydro_only_rev$Data_Split == "Test"])
table(pred_new_thresh,
      full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"],
      full_hydro_only_rev$us_num.f[full_hydro_only_rev$Data_Split == "Test"])

  ## US 1s only
 
pred_new_thresh_us1 = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 1],threshold = thresh)
table(pred_new_thresh_us1,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 1]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 1]))
table(pred_new_thresh_us1,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 1]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 1]))

pred_new_thresh_us2 = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 2],threshold = thresh)
table(pred_new_thresh_us2,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 2]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 2]))

pred_new_thresh_us12 = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f %in% c(1,2)],threshold = thresh)
table(pred_new_thresh_us12,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f %in% c(1,2)]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f %in% c(1,2)]))

pred_new_thresh_us3 = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 3],threshold = thresh)
table(pred_new_thresh_us3,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 3]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 3]))

pred_new_thresh_us4 = get_pred_class(full_hydro_only_rev$Scaled_Pred[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 4],threshold = thresh)
table(pred_new_thresh_us4,
      na.omit(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 4]),
      na.omit(full_hydro_only_rev$renal_scan1[full_hydro_only_rev$Data_Split == "Test" & full_hydro_only_rev$us_num.f == 4]))


53+28+11+6+1

na.omit(full_dat$study_id[full_dat$vcug_yn == 1 & full_dat$Data_Split == "Test"])
na.omit(full_dat$study_id[full_dat$vcug_yn == 0 & full_dat$Data_Split == "Test"])

  ###
  ###   GRAPHING THOSE WHO HAD A VCUG BUT DIDN'T GET SURGERY
  ###

vcug.set = full_hydro_only_rev[full_hydro_only_rev$Target == 0 & 
                                 full_hydro_only_rev$us_num %in% 1:3 & 
                                 (full_hydro_only_rev$renal_scan1 == "Yes" | 
                                    full_hydro_only_rev$renal_scan2 == "Yes" |
                                    full_hydro_only_rev$renal_scan3 == "Yes" ),]
str(vcug.set)

  ## Test only
stripchart(split(vcug.set$Scaled_Pred[vcug.set$Data_Split == "Test"],
                 vcug.set$us_num[vcug.set$Data_Split == "Test"]),
           method = "jitter",jitter = 0.05,vertical = TRUE,
           pch = 19, cex=0.5,
           col = c(rep("pink",nrow(vcug.set[vcug.set$Data_Split == "Test",])*0.5),
                   rep("blue",nrow(vcug.set[vcug.set$Data_Split == "Test",])*0.5)))
abline(h=thresh)

sum(vcug.set$Scaled_Pred[vcug.set$Data_Split == "Test"] < thresh)
sum(vcug.set$Scaled_Pred[vcug.set$Data_Split == "Test"] < 0.2)
length(unique(vcug.set$study_id[vcug.set$Data_Split == "Test"])) ## 26
length(unique(vcug.set$study_id[vcug.set$Data_Split == "Test" & vcug.set$Scaled_Pred < thresh])) ## 12
15/26

dim(full_dat[full_dat$Data_Split == "Test",])

###############################
###############################
#
#   Investigating AUC by Different variables
#
###############################
###############################
      ### 
      ###    SFU GRADE 
      ###

## Plot of data by SFU_grade 
ggplot(full_hydro_only_rev,aes(x = Target.f,y = Pred_val,col = SFU_grade)) + geom_point() + geom_jitter() + facet_grid(SFU_grade~Data_Split)
us_num_viol = ggplot(full_hydro_only_rev,aes(x = Target.f,y = Pred_val, fill = SFU_grade)) + geom_violin() + facet_grid(SFU_grade~Data_Split) + 
  labs(fill = "SFU grade") + xlab("Target") + ylab("Prediction") + 
  theme(strip.text.x = element_text(size = 15),
        axis.title = element_text(size = 15),
        axis.text.x = element_text(size = 13))
us_num_viol
ggplot_cols = unique(ggplot_build(us_num_viol)$data[[1]]$fill)
    
    ## extracting ggplot cols
ggplot(full_dat,aes(x = Target,y = Pred_val,col = factor(us_num))) + geom_point() + geom_jitter() + facet_grid(us_num~Data_Split)
us_num_viol = ggplot(full_dat,aes(x = factor(Target),y = Pred_val, fill = factor(us_num))) + geom_violin() + facet_grid(us_num~Data_Split) + 
  labs(fill = "US number") + xlab("Target") + ylab("Prediction") + 
  theme(strip.text.x = element_text(size = 15),
        axis.title = element_text(size = 15),
        axis.text.x = element_text(size = 13))
us_num_viol
ggplot_cols = unique(ggplot_build(us_num_viol)$data[[1]]$fill)

  ## TRAINING

get_auc = function(df,sfu_grade = NULL,data_split = "Test"){
  if(is.null(sfu_grade)){
    my_auc = auc(df$Target[df$Data_Split == data_split],
                 df$Scaled_Pred[df$Data_Split == data_split])  
  } else{
    my_auc = auc(df$Target[df$SFU_grade == sfu_grade & df$Data_Split == data_split],
                 df$Scaled_Pred[df$SFU_grade == sfu_grade & df$Data_Split == data_split])  
  }
  return(my_auc)
}

get_auprc = function(df,sfu_grade = NULL,data_split = "Training"){
  if(is.null(sfu_grade)){
    no.surg.scores = na.omit(df$Scaled_Pred[df$Target == 0 & full_hydro_only_rev$Data_Split == data_split])
    surg.scores = na.omit(df$Scaled_Pred[df$Target == 1 & full_hydro_only_rev$Data_Split == data_split])
  } else{
    no.surg.scores = na.omit(df$Scaled_Pred[df$Target == 0 & df$SFU_grade == sfu_grade & df$Data_Split == data_split])
    surg.scores = na.omit(df$Scaled_Pred[df$Target == 1 & df$SFU_grade == sfu_grade & df$Data_Split == data_split])
  }
  my_pr = pr.curve(scores.class0 = no.surg.scores,scores.class1 = surg.scores)
  return(my_pr$auc.integral)
}

for(split in c("Training","Test")){
  for(grade in 2:4){
    
    cat(split," Grade: ",grade," \tAUROC: ")
    
    cat(get_auc(df = full_hydro_only_rev,
            sfu_grade = grade,
            data_split = split))
    cat("\t AUPRC: ")
    cat(get_auprc(df = full_hydro_only_rev,
              sfu_grade = grade,
              data_split = split))
    cat("\n")
    
  }
}

cat("Training:\n")
cat(get_auc(df = full_hydro_only_rev,
            data_split = "Training"),"\n")
cat(get_auprc(df = full_hydro_only_rev,
              data_split = "Training"))
cat("Test:\n")
cat(get_auc(df = full_hydro_only_rev,
            data_split = "Test"),"\n")
cat(get_auprc(df = full_hydro_only_rev,
              data_split = "Test"))

for(split in c("Training","Test")){
  for(grade in 0:4){
    cat(split," Grade: ",grade,"\n")
    cat(table(full_hydro_only_rev$Target[full_hydro_only_rev$SFU_grade == grade & full_hydro_only_rev$Data_Split == split]),"\n")
  }
}

table(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Training"])
table(full_hydro_only_rev$Target[full_hydro_only_rev$Data_Split == "Test"])

##### AUC CURVE FOR EACH KIDNEY SIDE 
## TRAIN
ks_auc_df = data.frame(matrix(nrow=0,ncol=3))
kidney_sides = c("Left","Right")
for(side in kidney_sides){
  df = simple_roc(full_dat$Target[full_dat$Data_Split == "Training" & full_dat$kidney_side == side],
                  full_dat$Pred_val[full_dat$Data_Split == "Training" & full_dat$kidney_side == side])
  df$kidney_side = side
  
  ks_auc_df = rbind(ks_auc_df,df)
}
ks_auc_df$us_num = factor(ks_auc_df$kidney_side,levels = kidney_sides)

ggplot(ks_auc_df,aes(x = FPR,y = TPR,col = kidney_side)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("Left" = ggplot_cols[1],
                                "Right" = ggplot_cols[2]),
                     name = "Kidney Side") + 
  theme(axis.title = element_text(size = 15))

##  TEST
ks_auc_df = data.frame(matrix(nrow=0,ncol=3))
kidney_sides = c("Left","Right")
for(side in kidney_sides){
  df = simple_roc(full_dat$Target[full_dat$Data_Split == "Test" & full_dat$kidney_side == side],
                  full_dat$Pred_val[full_dat$Data_Split == "Test" & full_dat$kidney_side == side])
  df$kidney_side = side
  
  ks_auc_df = rbind(ks_auc_df,df)
}
ks_auc_df$us_num = factor(ks_auc_df$kidney_side,levels = kidney_sides)

ggplot(ks_auc_df,aes(x = FPR,y = TPR,col = kidney_side)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("Left" = ggplot_cols[1],
                                "Right" = ggplot_cols[2]),
                     name = "Kidney Side") + 
  theme(axis.title = element_text(size = 15))

##### AUC CURVE FOR EACH ETIOLOGY 
## TRAIN
ks_auc_df = data.frame(matrix(nrow=0,ncol=3))
kidney_sides = c("Left","Right")
for(side in kidney_sides){
  df = simple_roc(full_dat$Target[full_dat$Data_Split == "Training" & full_dat$kidney_side == side],
                  full_dat$Pred_val[full_dat$Data_Split == "Training" & full_dat$kidney_side == side])
  df$kidney_side = side
  
  ks_auc_df = rbind(ks_auc_df,df)
}
ks_auc_df$us_num = factor(ks_auc_df$kidney_side,levels = kidney_sides)

ggplot(ks_auc_df,aes(x = FPR,y = TPR,col = kidney_side)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("Left" = ggplot_cols[1],
                                "Right" = ggplot_cols[2]),
                     name = "Kidney Side") + 
  theme(axis.title = element_text(size = 15))

##  TEST
ks_auc_df = data.frame(matrix(nrow=0,ncol=3))
kidney_sides = c("Left","Right")
for(side in kidney_sides){
  df = simple_roc(full_dat$Target[full_dat$Data_Split == "Test" & full_dat$kidney_side == side],
                  full_dat$Pred_val[full_dat$Data_Split == "Test" & full_dat$kidney_side == side])
  df$kidney_side = side
  
  ks_auc_df = rbind(ks_auc_df,df)
}
ks_auc_df$us_num = factor(ks_auc_df$kidney_side,levels = kidney_sides)

ggplot(ks_auc_df,aes(x = FPR,y = TPR,col = kidney_side)) + 
  geom_line(size=1.5) + 
  theme_bw() + 
  scale_color_manual(values = c("Left" = ggplot_cols[1],
                                "Right" = ggplot_cols[2]),
                     name = "Kidney Side") + 
  theme(axis.title = element_text(size = 15))

##### AUC CURVE FOR EACH US NUMBER
## TRAIN
us_auc_df = data.frame(matrix(nrow=0,ncol=4))
us_num_levels = 1:7
for(us_num in us_num_levels){
  if(length(full_dat$Target[full_dat$Data_Split == "Training" & full_dat$us_num == us_num]) > 0){
    df = simple_roc(full_dat$Target[full_dat$Data_Split == "Training" & full_dat$us_num == us_num],
                    full_dat$Pred_val[full_dat$Data_Split == "Training" & full_dat$us_num == us_num])
    df$us_num = us_num
    
    us_auc_df = rbind(us_auc_df,df) 
  }
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


    #######################
    ## FULL DATA GRAPHS
    #######################
## GRAPHS ON HYDRO ONLY DATASET
ggplot(full_scaled2,aes(x = Target.f,y = Scaled_Pred,col = Data_Split)) + geom_point() + geom_jitter() + facet_grid(.~Data_Split) + ylim(c(0,1))

ggplot(full_scaled2,aes(x = Target.f,y = Scaled_Pred,col = gender)) + geom_point() + geom_jitter() + facet_grid(gender~Data_Split) + ylim(c(0,1))
ggplot(full_scaled2,aes(x = Target.f,y = Scaled_Pred,col = manu)) + geom_point() + geom_jitter() + facet_grid(manu~Data_Split) + ylim(c(0,1))
ggplot(full_scaled2,aes(x = Target.f,y = Scaled_Pred,col = kidney_side)) + geom_point() + geom_jitter() + facet_grid(kidney_side~Data_Split) + ylim(c(0,1))
ggplot(full_scaled2,aes(x = Target.f,y = Scaled_Pred,col = us_num.f)) + geom_point() + geom_jitter() + facet_grid(us_num.f~Data_Split) + ylim(c(0,1))

  ## SORTED DATA GRAPHS
full_scaled2.sorted = full_scaled2[order(full_scaled2$date_of_current_us.date),]
ggplot(full_scaled2.sorted,aes(x = age_at_us,y = Pred_val,group = study_id)) + geom_line() + facet_grid(Target.f ~ Data_Split)

  ###############
  ### FITTING GP TO THE HYDRO only DATA
  ###############
library(GPfit)

train_hydro_only_no_surg = full_hydro_only_rev[full_hydro_only_rev$Data_Split == "Training" & full_hydro_only_rev$Target == 0,]

gp_train_hydro_only_no_surg = GP_fit(na.omit((train_hydro_only_no_surg$age_at_us - train_hydro_only_no_surg$age_at_baseline)/max(na.omit(train_hydro_only_no_surg$age_at_us))),
                                  train_hydro_only_no_surg$Scaled_Pred[!is.na(train_hydro_only_no_surg$age_at_us)])

plot(gp_train_hydro_only_no_surg)


## SAME INDS IN EACH SET 
## COUNT NUMBER OF ULTRASOUNDS FOR EACH PERSON 
  ## NUMBER OF ULTRASOUNDS REMOVED FOR QC (NUM ZEROS PER TABLE(TABLE(...)))
    ## \/ add this for each ID -- can make table
sum(ifelse(table(data_triad$train$kidney_side[data_triad$train$study_id == 339],
                 data_triad$train$us_num[data_triad$train$study_id == 339]) == 0,
           yes = 1, no = 0))




##############
#### OLD CODE 
##############

## MAKE DATASET OF HYDRO KIDNEYS ONLY: 
  ## INCLUDE SFU GRADE, VCUG Y/N, 
str(full_dat)
str(phn.raw)
names(phn.raw)

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

  ## CREATING SCALED PREDICTIONS



### GETTING MIS-DIAGNOSED PATIENTS
# hydro_only_full_dat[hydro_only_full_dat$Target == 1 & 
#                       hydro_only_full_dat$Data_Split == "Test" & 
#                       hydro_only_full_dat$surgery_type == "Circumcision",
#                     c("study_id","us_num","kidney_side","surgery_type")]

# hydro_only_full_dat[hydro_only_full_dat$Target == 1 & 
#                       hydro_only_full_dat$Data_Split == "Training" & 
#                       hydro_only_full_dat$surgery_type == "Circumcision",
#                     c("study_id","us_num","kidney_side","surgery_type","surgery_type2")]


#hydro_only_full_dat[hydro_only_full_dat$Target == 0 & hydro_only_full_dat$Data_Split == "Test" & hydro_only_full_dat$Pred_val > 0.6,c("study_id","us_num","kidney_side")]


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



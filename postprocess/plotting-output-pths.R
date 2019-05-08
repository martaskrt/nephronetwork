setwd("C:/Users/Lauren/Desktop/DS Core/Projects/Urology/eval_nn/summary_files/")

### PACKAGES

library("ggplot2")

### USER-DEFINED FUCTIONS

get_manu = function(full_id_vec){
  manu = unlist(lapply(strsplit(full_id_vec,split = "_"),function(x){x[length(x)]}))
  return(manu)
}

### PROCESSING THE DATA

analysis_name = "unet_20190503_vanilla_CV_lr0.001_e35_bs256_c1_SGD_pi"

train = read.csv(paste0(analysis_name,"_train.csv"),header=TRUE,as.is=TRUE)
val = read.csv(paste0(analysis_name,"_val.csv"),header=TRUE,as.is=TRUE)
test = read.csv(paste0(analysis_name,"_test.csv"),header=TRUE,as.is=TRUE)
head(train)

data_triad = list("train" = train,"val" = val,"test" = test)

data_triad = lapply(data_triad,function(x){x$manu = get_manu(x[,"full_ID"]) ; return(x)})
str(data_triad)

full_dat = Reduce(rbind,data_triad)
full_dat$set = c(rep("train",nrow(data_triad[["train"]])),
                 rep("val",nrow(data_triad[["val"]])),
                 rep("test",nrow(data_triad[["test"]])))
full_dat$Data_Split = factor(full_dat$set,levels = c("train","val","test"))


table(full_dat$kidney_side)

ggplot(full_dat,aes(x = Pred_val,y = Target,size = age_at_baseline)) + geom_point() + facet_grid(.~Data_Split)



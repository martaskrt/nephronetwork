
## LIBRARIES
library(rjson)
library(mltools)
library(ggplot2)


## WD
setwd("C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/csv/")


## FUNCTIONS
get_auc_ci = function(true_vals, pred_vals, n_its){
  
  true_auc = mltools::auc_roc(preds = pred_vals, actuals = true_vals)
  n_data = length(pred_vals)
  
  resampled_aucs = sapply(1:n_its, function(i){
    samp_idx = sample(1:n_data, size = n_data, replace=TRUE)
    auc_out = mltools::auc_roc(pred_vals[samp_idx],actuals = true_vals[samp_idx])
    return(auc_out)
  })
  
  sorted_aucs = sort(resampled_aucs)
  ci = quantile(sorted_aucs,c(0.025, 0.975))
  
  out_list = list("auc" = true_auc, "ci" = ci)
  return(out_list)
}

create_auc_test_df = function(in_list, num_its=500){
  
  list_df = data.frame(matrix(nrow = 0, ncol = 5))
  names(list_df) = c("method","epoch","auc","lcl","ucl")
  
  for(i in 1:length(in_list)){
    if(!is.null(names(in_list))){
      method_name = names(in_list)[i]
    } else{
      method_name = paste0("method_",i)
    }
    
    for(epoch in 1:length(in_list[[i]])){
      auc_list = get_auc_ci(true_vals = in_list[[i]][[epoch]]$labels,
                            in_list[[i]][[epoch]]$pred_vals,
                            n_its = num_its)
      auc_val = auc_list$auc
      my_lcl = auc_list$ci[1]
      my_ucl = auc_list$ci[2]
      
      my_row = data.frame(method=method_name, epoch, auc=auc_val, lcl=my_lcl, ucl=my_ucl)
      
      list_df = rbind(list_df,my_row)
    }
    
  }
  
  return(list_df)
}


### TARGET = S3000

### STANFORD + SICKKIDS
nomsda_stsk = fromJSON(file = "noMSDA_SIEMENS_S3000_0_test_Detailed_st_sk_n500.json")
cmsda_stsk = fromJSON(file = "cMSDA_SIEMENS_S3000_0_test_Detailed_st_sk_n500.json")
msda_stsk = fromJSON(file = "MSDA_SIEMENS_S3000_11_test_Detailed_st_sk_n500.json")
stsk_list = list("noM3SDA" = nomsda_stsk, "cM3SDA" = cmsda_stsk, "M3SDA" = msda_stsk)


### STANFORD ONLY -- BS4
nomsda_st_bs4 = fromJSON(file = "noMSDA_SIEMENS_S3000_0_test_Detailed_st_n500_15ep_bs4.json")
cmsda_st_bs4 = fromJSON(file = "cMSDA_SIEMENS_S3000_1_test_Detailed_st_n500_15ep_bs4.json")
msda_st_bs4 = fromJSON(file = "MSDA_SIEMENS_S3000_16_test_Detailed_st_n500_15ep_bs4.json")
st_list_bs4 = list("noM3SDA" = nomsda_st_bs4,
               'cM3SDA' = cmsda_st_bs4,
               "M3SDA" = msda_st_bs4)

### STANFORD ONLY -- BS8
nomsda_st_bs8 = fromJSON(file = "noMSDA_SIEMENS_S3000_0_test_Detailed_st_n500_15ep_bs8.json")
cmsda_st_bs8 = fromJSON(file = "cMSDA_SIEMENS_S3000_0_test_Detailed_st_n500_15ep_bs8.json")
msda_st_bs8 = fromJSON(file = "MSDA_SIEMENS_S3000_11_test_Detailed_st_n500_15ep_bs8.json")
st_list_bs8 = list("noM3SDA" = nomsda_st_bs8, "cM3SDA" = cmsda_st_bs8, "M3SDA" = msda_st_bs8)

### STANFORD ONLY -- BS16
nomsda_st_bs16 = fromJSON(file = "noMSDA_SIEMENS_S3000_0_test_Detailed_st_n500_bs16.json")
cmsda_st_bs16 = fromJSON(file = "cMSDA_SIEMENS_S3000_0_test_Detailed_st_n500_bs16.json")
msda_st_bs16 = fromJSON(file = "MSDA_SIEMENS_S3000_11_test_Detailed_st_n500_bs16.json")
st_list_bs16 = list("noM3SDA" = nomsda_st_bs16, "cM3SDA" = cmsda_st_bs16, "M3SDA" = msda_st_bs16)


### STANFORD + SICKKIDS, INSTITUTION DOMAINS
nomsda_stsk_inst = fromJSON(file = "noMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs4.json")
cmsda_stsk_inst = fromJSON(file = "cMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs4.json")
msda_stsk_inst = fromJSON(file = "MSDA_Stanford_S3000_1_test_Detailed_sk_st_inst_n500_15ep_bs4.json")
stsk_inst_list_bs4 = list("noM3SDA" = nomsda_stsk_inst, "cM3SDA" = cmsda_stsk_inst, "M3SDA" = msda_stsk_inst)

### STANFORD + SICKKIDS, INSTITUTION DOMAINS -- BS8
nomsda_stsk_inst_bs8 = fromJSON(file = "noMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs8.json")
# cmsda_stsk_inst_bs8 = fromJSON(file = "cMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs8.json")
cmsda_stsk_inst_bs8 = fromJSON(file = "cMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs8_v2.json")
msda_stsk_inst_bs8 = fromJSON(file = "MSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs8.json")
stsk_inst_list_bs8 = list("noM3SDA" = nomsda_stsk_inst_bs8, "cM3SDA" = cmsda_stsk_inst_bs8, "M3SDA" = msda_stsk_inst_bs8)


### STANFORD + SICKKIDS, INSTITUTION DOMAINS -- BS16
nomsda_stsk_inst_bs16 = fromJSON(file = "noMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs16.json")
cmsda_stsk_inst_bs16 = fromJSON(file = "cMSDA_Stanford_S3000_1_test_Detailed_sk_st_inst_n500_15ep_bs16.json")
msda_stsk_inst_bs16 = fromJSON(file = "MSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs16.json")
stsk_inst_list_bs16 = list("noM3SDA" = nomsda_stsk_inst_bs16, "cM3SDA" = cmsda_stsk_inst_bs16, "M3SDA" = msda_stsk_inst_bs16)


#### **** FINISH THIS
### STANFORD + SICKKIDS, INSTITUTION DOMAINS, NO DISCR -- BS16
nomsda_stsk_inst_bs16_nodiscr = fromJSON(file = "noMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs16_nodiscr.json")
cmsda_stsk_inst_bs16_nodiscr = fromJSON(file = "cMSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs16_nodiscr.json")
msda_stsk_inst_bs16_nodiscr = fromJSON(file = "MSDA_Stanford_S3000_0_test_Detailed_sk_st_inst_n500_15ep_bs16_nodiscr.json")
stsk_inst_list_bs16_nodiscr = list("noM3SDA" = nomsda_stsk_inst_bs16_nodiscr, "cM3SDA" = cmsda_stsk_inst_bs16_nodiscr, "M3SDA" = msda_stsk_inst_bs16_nodiscr)


stsk_inst_vs_stonly_bs8 = list("nomsda_stsk_inst_bs8" = nomsda_stsk_inst_bs8, 
                          "cmsda_stsk_inst_bs8" = cmsda_stsk_inst_bs8, 
                          "msda_stsk_inst_bs8" = msda_stsk_inst_bs8,
                          "nomsda_st_bs8" = nomsda_st_bs8, 
                          "cmsda_st_bs8" = cmsda_st_bs8, 
                          "msda_st_bs8" = msda_st_bs8)


### CREATE GRAPHS


theme_set(
  theme_classic(base_size = 20)
)

  ## STSK
stsk_df = create_auc_test_df(stsk_list)
head(stsk_df)

ggplot(stsk_df,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.5,1.0)


  ## ST ONLY --- BS 4, 8, 16
st_df_bs4 = create_auc_test_df((st_list_bs4))
st_df_bs8 = create_auc_test_df((st_list_bs8))
st_df_bs16 = create_auc_test_df((st_list_bs16))

st_df_bs4$BatchSize = 4
st_df_bs8$BatchSize = 8
st_df_bs16$BatchSize = 16

st_df = rbind(st_df_bs4,st_df_bs8,st_df_bs16)

ggplot(st_df,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0) + facet_grid(~BatchSize)


  ## STSK INSTITUTION DOMAINS: SOURCES: SK, ST(AS,GE) ; TARGET: ST(S3000) -- BS4
stsk_inst_df = create_auc_test_df(stsk_inst_list_bs4)
head(stsk_inst_df)

ggplot(stsk_inst_df,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0)



## STSK INSTITUTION DOMAINS: SOURCES: SK, ST(AS,GE) ; TARGET: ST(S3000) -- BS8
stsk_inst_df_bs8 = create_auc_test_df(stsk_inst_list_bs8)
head(stsk_inst_df_bs8)

ggplot(stsk_inst_df_bs8,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.85,1.0)



## STSK INSTITUTION DOMAINS: SOURCES: SK, ST(AS,GE) ; TARGET: ST(S3000) -- BS16
stsk_inst_df_bs16 = create_auc_test_df(stsk_inst_list_bs16)
head(stsk_inst_df_bs16)

ggplot(stsk_inst_df_bs16,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0)

## STSK INSTITUTION DOMAINS: SOURCES: SK, ST(AS,GE) ; TARGET: ST(S3000) -- BS4, 8, 16
stsk_inst_df$BatchSize = 4
stsk_inst_df_bs8$BatchSize = 8
stsk_inst_df_bs16$BatchSize = 16
stsk_inst_df_bs4_8_16 = rbind(stsk_inst_df, stsk_inst_df_bs8, stsk_inst_df_bs16)

ggplot(stsk_inst_df_bs4_8_16,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0) + facet_grid(~BatchSize)


## STSK INSTITUTION DOMAINS ; NO DISCR -- BS16
stsk_inst_df_bs16_nodiscr = create_auc_test_df(stsk_inst_list_bs16_nodiscr)
head(stsk_inst_df_bs16_nodiscr)

ggplot(stsk_inst_df_bs16_nodiscr,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0)


## STSK INSTITUTION DOMAINS W/ vs W/O DISCR -- BS16
stsk_inst_df_bs16$Concordance = "Concordance"
stsk_inst_df_bs16_nodiscr$Concordance = "No Concordance"

stsk_inst_df_bs16_wwodiscr = rbind(stsk_inst_df_bs16, stsk_inst_df_bs16_nodiscr)

ggplot(stsk_inst_df_bs16_wwodiscr,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.8,1.0) + facet_grid(~Concordance)

## ST ONLY -- BS8
st_df_bs8 = create_auc_test_df(st_list_bs8)
head(st_df_bs8)

ggplot(st_df_bs8,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.85,1.0)


## STSK INSTITUTION DOMAINS VS ST ONLY -- BS8
stsk_inst_vs_stonly_bs8_df = create_auc_test_df(stsk_inst_vs_stonly_bs8)
head(stsk_inst_vs_stonly_bs8_df)

ggplot(stsk_inst_vs_stonly_bs8_df,aes(x = epoch, y = auc, col = method)) + geom_line() + 
  geom_point() + 
  geom_errorbar(aes(ymin=lcl, ymax=ucl)) + ylim(0.85,1.0)


### NEXT STEPs: 
##    1. SET UP TO RUN INFERENCE ON TEST SET WITH MODEL SAVED AT xx EPOCH
##    2. INCORPORATE CHOP DATA
##    3. CONFIRM METHDOS + RESULTS W/ANNA AND BO FRIDAY



### CREATE GRAPH OF UMAP OF FEATURES OVER EPOCHS -- MAYBE SHOW SHIFT? 
  ##  COLOR BY CASE/CONTROL, SHAPE BY SOURCE, OUTLINE TARGET




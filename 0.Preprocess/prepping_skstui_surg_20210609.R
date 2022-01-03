
library(readxl)
library(reshape2)
library(rjson)
library(RJSONIO)

### 
###    SICKKIDS DATA
###

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SickKids_wST_filenames_20210216.json")

library(V8)
test = v8()
test$assign("dat", JS(raw_in))
sk_list = test$get("dat")

length(sk_list)
names(sk_list)

sk_list[["STID43"]]

### get BL age groups here 

### 
###    STANFORD DATA
###

stan_dat = read_excel("C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/Datasheets/AI Hydronephrosis Data deidentified.xlsx")
head(stan_dat)

stan_dat$l_surg = 0
stan_dat$l_surg[stan_dat$`Did you indicate surgery?` == "Yes" & stan_dat$`Which kidney was sx indicated for? (L/R)` == "Left"] = 1

stan_dat$r_surg = 0
stan_dat$r_surg[stan_dat$`Did you indicate surgery?` == "Yes" & stan_dat$`Which kidney was sx indicated for? (L/R)` == "Right"] = 1

stan_dat$`Did you indicate surgery? Binary` = ifelse(stan_dat$`Did you indicate surgery?` == "Yes", 1, 0)

stan_dat$`Did you indicate surgery at any time?` = stan_dat$`Did you indicate surgery? Binary`

for(i in 1:nrow(stan_dat)){
  stan_dat$`Did you indicate surgery at any time?`[i] = max(stan_dat$`Did you indicate surgery? Binary`[stan_dat$anon_mrn == stan_dat$anon_mrn[i]])
}
table(stan_dat$`Did you indicate surgery at any time?`)

stan_dat$l_surg_anytime = 0
stan_dat$l_surg_anytime[stan_dat$`Did you indicate surgery at any time?` == 1 & stan_dat$`side of hydronephrosis` == "Left"] = 1

stan_dat$r_surg_anytime = 0
stan_dat$r_surg_anytime[stan_dat$`Did you indicate surgery at any time?` == 1 & stan_dat$`side of hydronephrosis` == "Right"] = 1

table(stan_dat$r_surg_anytime)
table(stan_dat$l_surg_anytime)



st_sk_list = sk_list

for(id in unique(stan_dat$anon_mrn)){
  df_sub = stan_dat[stan_dat$anon_mrn == id,]
  
  st_sk_list[[id]] = list("Left" = list(), "Right" = list())
  
  # print(unique(df_sub$`Did you indicate surgery?`))
  
  for(acc in df_sub$anon_accession){
    if(dir.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-",acc))){
      st_sk_list[[id]][["Left"]][["surgery"]] = max(df_sub$l_surg_anytime[df_sub$anon_mrn == id])
      st_sk_list[[id]][["Right"]][["surgery"]] = max(df_sub$r_surg_anytime[df_sub$anon_mrn == id])
      st_sk_list[[id]][["Sex"]] = ifelse(df_sub$sex[df_sub$anon_mrn == id][1] == "M", 1, 2)
      st_sk_list[[id]][["BL_date"]] = "2021-01-01"
      
      if(file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LS_cropped.jpg")) &
         file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LT_cropped.jpg"))){
        st_sk_list[[id]][["Left"]][[acc]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LS_cropped.jpg")
        st_sk_list[[id]][["Left"]][[acc]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LT_cropped.jpg")
        st_sk_list[[id]][["Left"]][[acc]][["US_machine"]] = "Stanford"
        st_sk_list[[id]][["Left"]][[acc]][["ApD"]] = "NA"
        st_sk_list[[id]][["Left"]][[acc]][["SFU"]] = "NA"
        st_sk_list[[id]][["Left"]][[acc]][["Age_wks"]] = 52*df_sub$age[df_sub$anon_accession == acc]
      } else if(file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LS_cropped.png")) &
                file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LT_cropped.png"))){
        st_sk_list[[id]][["Left"]][[acc]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LS_cropped.png")
        st_sk_list[[id]][["Left"]][[acc]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/LT_cropped.png")
        st_sk_list[[id]][["Left"]][[acc]][["US_machine"]] = "Stanford"
        st_sk_list[[id]][["Left"]][[acc]][["ApD"]] = "NA"
        st_sk_list[[id]][["Left"]][[acc]][["SFU"]] = "NA"
        st_sk_list[[id]][["Left"]][[acc]][["Age_wks"]] = 52*df_sub$age[df_sub$anon_accession == acc]
      }
      
      if(file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RS_cropped.jpg")) &
         file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RT_cropped.jpg"))){
        st_sk_list[[id]][["Right"]][[acc]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RS_cropped.jpg")
        st_sk_list[[id]][["Right"]][[acc]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RT_cropped.jpg")
        st_sk_list[[id]][["Right"]][[acc]][["US_machine"]] = "Stanford"
        st_sk_list[[id]][["Right"]][[acc]][["ApD"]] = "NA"
        st_sk_list[[id]][["Right"]][[acc]][["SFU"]] = "NA"
        st_sk_list[[id]][["Right"]][[acc]][["Age_wks"]] = 52*df_sub$age[df_sub$anon_accession == acc]
        
      } else if(file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RS_cropped.png")) &
                file.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RT_cropped.png"))){
        st_sk_list[[id]][["Right"]][[acc]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RS_cropped.png")
        st_sk_list[[id]][["Right"]][[acc]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/RLST_labeled_imgs/batch_dicom_",id,"-", acc, "/cropped/RT_cropped.png")
        st_sk_list[[id]][["Right"]][[acc]][["US_machine"]] = "Stanford"
        st_sk_list[[id]][["Right"]][[acc]][["ApD"]] = "NA"
        st_sk_list[[id]][["Right"]][[acc]][["SFU"]] = "NA"
        st_sk_list[[id]][["Right"]][[acc]][["Age_wks"]] = 52*df_sub$age[df_sub$anon_accession == acc]
        
      }
      
    }
  }
}


st_sk_list[["SU2bae87d"]]

length(st_sk_list)

st_sk_list[["SU2bae8a1"]]

st_sk_list[["SU2bae87d"]][["Left"]]
st_sk_list[["SU2bae883"]][["Left"]][["surgery"]]
st_sk_list[["SU2bae8a1"]]

### 
###    UIOWA DATA
###

ui_list = list()

ui_dat = read.csv("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UIowa_Datasheet2.csv",header=TRUE,as.is=TRUE)
head(ui_dat)
str(ui_dat)

ui_dat$age_wks = difftime(as.Date(ui_dat$Ultrasound.Date),as.Date(ui_dat$DOB),units = "week")

for(id in ui_dat$Name){

  if(dir.exists(paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/HN AI Images/RLST_Labeled_Images/",id))){
    
    ui_list[[id]] = list()
    
    if(ui_dat$U.B[ui_dat$Name == id] == "L"){
      ui_list[[id]][["Left"]] = list()
      ui_list[[id]][["Left"]][["surgery"]] = ifelse(substr(id,1,1) == "O", 1, 0)
      ui_list[[id]][["Left"]][["1"]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/HN AI Images/RLST_Labeled_Images/",id,"/LS_cropped.png")
      ui_list[[id]][["Left"]][["1"]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/HN AI Images/RLST_Labeled_Images/",id,"/LT_cropped.png")
      ui_list[[id]][["Left"]][["1"]][["Age_wks"]] = ui_dat$age_wks[ui_dat$Name == id]
      ui_list[[id]][["Left"]][["1"]][["US_machine"]] = "UIowa"
      ui_list[[id]][["Left"]][["1"]][["ApD"]] = "NA"
      ui_list[[id]][["Sex"]] = ui_dat$Gender[ui_dat$Name == id]
      
    } else if(ui_dat$U.B[ui_dat$Name == id] == "R"){
      ui_list[[id]][["Right"]] = list()
      ui_list[[id]][["Right"]][["surgery"]] = ifelse(substr(id,1,1) == "O", 1, 0)
      ui_list[[id]][["Right"]][["1"]][["sag"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/HN AI Images/RLST_Labeled_Images/",id,"/RS_cropped.png")
      ui_list[[id]][["Right"]][["1"]][["trv"]] = paste0("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/HN AI Images/RLST_Labeled_Images/",id,"/RT_cropped.png")
      ui_list[[id]][["Right"]][["1"]][["Age_wks"]] = ui_dat$age_wks[ui_dat$Name == id]
      ui_list[[id]][["Right"]][["1"]][["US_machine"]] = "UIowa"
      ui_list[[id]][["Right"]][["1"]][["ApD"]] = "NA"
      ui_list[[id]][["Sex"]] = ui_dat$Gender[ui_dat$Name == id]
      
    } else{
      cat("unknown kidney laterality")
    }
  }
  
}

names(ui_list)

lapply(ui_list,function(x){na.omit(c(x$Right$surgery,x$Left$surgery))})

###
###     INPUT FOR NEW SURGERY MODEL -- JUNE 2021
###

write(toJSON(ui_list), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIonly_filenames_20210612.json")


###
###     INPUT FOR CURRENT SURGERY MODEL
###

# write(toJSON(st_sk_list), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SK_ST_filenames_20210223.json")

###
###     STANFORD ONLY INPUT FOR CURRENT SURGERY MODEL
###

stan_only_list = st_sk_list[names(st_sk_list)[grep(pattern = "SU", x = names(st_sk_list))]]
names(stan_only_list)
stan_only_list[["SU2bae87d"]]
write(toJSON(stan_only_list), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_StanOnly_filenames_20210612.json")

    ###     CREATING SUBSET FOR FINE-TUNING

stan_dat$`Did you indicate surgery at any time?`

## n = 112
length(unique(stan_dat$anon_mrn))

stan_sub = stan_dat[!duplicated(stan_dat$anon_mrn),]

table(stan_sub$`Did you indicate surgery at any time?`)

set.seed(1234)
test_ids = c(stan_sub$anon_mrn[stan_sub$`Did you indicate surgery at any time?` == 1][sample.int(n = 12, size = 6)],
             stan_sub$anon_mrn[stan_sub$`Did you indicate surgery at any time?` == 0][sample.int(n = 100, size = 30)])

train_ids = stan_sub$anon_mrn[!(stan_sub$anon_mrn %in% test_ids)]

stan_train = stan_only_list[train_ids]
stan_test = stan_only_list[test_ids]

write(toJSON(stan_train), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_StanOnlyTrain_filenames_20210427.json")
write(toJSON(stan_test), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_StanOnlyTest_filenames_20210427.json")

#####**
###
###     CREATING INPUT FOR SDA VERSION 
###
#####**

name_vec = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","machine","test","age_wks","side")
st_bl_dat = data.frame(matrix(nrow = 0, ncol = length(name_vec)))

## Adding Stanford data
for(id in unique(stan_dat$anon_mrn)){
  df_sub = stan_dat[stan_dat$anon_mrn == id,]
  for(acc in df_sub$anon_accession){
    for(side in c("Left","Right")){
      
      # cat("\nid:\n")
      # cat(id)
      # cat("\nacc:\n")
      # cat(acc)
      # cat("\nside:\n")
      # cat(side)
      
      if(length(st_sk_list[[id]][[side]][[acc]][["sag"]]) > 0){
        add_df = data.frame(SAG_FILE = st_sk_list[[id]][[side]][[acc]][["sag"]],
                            TRV_FILE = st_sk_list[[id]][[side]][[acc]][["trv"]], 
                            IMG_ID = paste0(id, "_", side, "_", acc),
                            surgery = max(ifelse(side == "Left" , df_sub$l_surg_anytime, df_sub$r_surg_anytime)),
                            machine = st_sk_list[[id]][[side]][[acc]][["US_machine"]],
                            test = NA, 
                            age_wks = st_sk_list[[id]][[side]][[acc]][["Age_wks"]],
                            side = side)
        st_bl_dat = rbind(st_bl_dat, add_df)          
      }
    }
  }
}

## Adding SickKids data
for(id in names(sk_list)){
  for(side in c("Left","Right")){
    us_nums = names(sk_list[[id]][[side]])[names(sk_list[[id]][[side]]) != "surgery"]
    surgery = sk_list[[id]][[side]][["surgery"]]
    for(us_num in us_nums){
      
      # cat("\nid:\n")
      # cat(id)
      # cat("\nus_num:\n")
      # cat(us_num)
      # cat("\nside:\n")
      # cat(side)
      
      
      if(length(sk_list[[id]][[side]][[us_num]][["sag"]]) > 0 & length(surgery) > 0 & length(sk_list[[id]][[side]][[us_num]][["trv"]]) > 0){
        add_df = data.frame(SAG_FILE = sk_list[[id]][[side]][[us_num]][["sag"]],
                            TRV_FILE = sk_list[[id]][[side]][[us_num]][["trv"]],
                            IMG_ID = paste0(id, "_", side, "_", us_num),
                            surgery = surgery,
                            machine = sk_list[[id]][[side]][[us_num]][["US_machine"]],
                            test = NA,
                            age_wks = ifelse(is.null(sk_list[[id]][[side]][[us_num]][["Age_wks"]]),90,sk_list[[id]][[side]][[us_num]][["Age_wks"]]),
                            side = side)
        st_bl_dat = rbind(st_bl_dat, add_df)
      }
    }
  }
}

stsk_df = st_bl_dat

table(stsk_df$machine,stsk_df$surgery)

orig_machines = c("acuson","atl","ge-healthcare","ge-medical-systems","philips-medical-systems","samsung-medison-co-ltd","siemens","toshiba-mec","toshiba-mec-us","OR_unknown")

    ###
    ### Stanford test
    ###

###
### No Stanford data training
###

st_test = st_bl_dat
st_test$test = 0
st_test = st_test[complete.cases(st_test),]

st_test$st_machine = paste0(substr(st_test$IMG_ID,1,2),"_unknown") 
st_test$st_machine[st_test$machine %in% c("Stanford", "SU_uknown")] = "Stanford"
st_test$st_machine[substr(st_test$machine,nchar(st_test$machine)-1,nchar(st_test$machine)) =="ST"] = "SKSilentTrial"
st_test$st_machine[st_test$st_machine == "ST_unknown"] = "SKSilentTrial"
st_test$st_machine[st_test$machine %in% orig_machines] = "SKOrig"

st_test$id = unlist(lapply(strsplit(st_test$IMG_ID,"_"), function(x)x[1]))
st_test$us_num = unlist(lapply(strsplit(st_test$IMG_ID,"_"), function(x)x[3]))
st_test$age_wks = as.numeric(st_test$age_wks)

set.seed(1234)
for(machine in unique(st_test$st_machine)){
  if(machine == "Stanford"){  # split by id @ LAUREN
    test_ids = sample(unique(st_test$id[st_test$st_machine == "Stanford"]), length(unique(st_test$id[st_test$st_machine == "Stanford"])) - 10)
  } else{
    test_ids = sample(unique(st_test$id[st_test$st_machine == machine]), round(0.15*length(unique(st_test$id[st_test$st_machine == machine]))))
  }
  st_test$test[st_test$id %in% test_ids] = 1
}

head(st_test)

st_test$source = st_test$st_machine

### imputing age at US
table(st_test$us_num, is.na(as.numeric(st_test$age_wks)))

plot(st_test$age_wks, st_test$us_num)

for(my_row in 1:nrow(st_test)){
  if(is.na(st_test$age_wks[my_row])){
    cat("\nTRUE 1\n")
    if(st_test$us_num[my_row] == 1){
      cat("\nTRUE 2\n")
      st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == 1]))
    } else{
      cat("\nTRUE 3\n")
      my_id = st_test$id[my_row]
      my_us_num = st_test$us_num[my_row]
      cat("\nmy_us_num\n")
      cat(my_us_num)
      df_sub = st_test[st_test$id == my_id,]
      
      if(any(as.numeric(df_sub$us_num) < my_us_num & !is.na(df_sub$age_wks))){
        cat("\nTRUE 4\n")
        comp_idx = which(df_sub$us_num == max(df_sub$us_num[df_sub$us_num < my_us_num]))[1]
        comp_df = df_sub[comp_idx,]
        comp_us_num =comp_df$us_num
        cat("\ncomp_us_num\n")
        cat(comp_us_num)       
        
        mean_diff = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num])) - mean(na.omit(st_test$age_wks[st_test$us_num == comp_us_num]))
        
        st_test$age_wks[my_row] = comp_df$age_wks + mean_diff
      } else{
        cat("\nTRUE 5\n")
        st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num]))        
      }
    }    
  }
}

# st_test[is.na(st_test$age_wks),]

for(my_row in 1:nrow(st_test)){
  if(is.na(st_test$age_wks[my_row])){
    cat("\nTRUE 5\n")
    st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num]))        
  }
}


st_test = st_test[,c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")]
st_test$surgery = as.numeric(st_test$surgery)

### balance set

table(st_test$source, st_test$test)

max_n = max(c(table(st_test$source, st_test$test)))

# name_vec = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
balanced_st_dat = st_test[complete.cases(st_test),]
b_sample = sample(st_test$IMG_ID[st_test$source == "SKSilentTrial" & st_test$test == 0],max_n-1168)
s_sample = sample(st_test$IMG_ID[st_test$source == "Stanford" & st_test$test == 0],max_n-10, replace = TRUE)
balanced_st_dat = rbind(balanced_st_dat, st_test[st_test$IMG_ID %in% b_sample,])
balanced_st_dat = rbind(balanced_st_dat, st_test[match(s_sample, st_test$IMG_ID),])

table(balanced_st_dat$source,balanced_st_dat$test)

table(is.na(as.numeric(balanced_st_dat$age_wks)),balanced_st_dat$source)

balanced_st_dat$side = ifelse(balanced_st_dat$side == "Left", 0, 1)

# write.csv(balanced_st_dat, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/surgery_sda_sksub_stsub_20210226.csv",quote=FALSE,row.names = FALSE)


make_unique = function(ids_in){
  # browser()
  
  ids_out = c()
  
  iter = 0
  for(id in ids_in){
    if(id %in% unique(ids_in[duplicated(ids_in)])){
      id = paste0(id,"_",iter)
      iter=iter+1
    }
    ids_out = c(ids_out,id)
  }
  
  return(ids_out)
}

make_balanced_data = function(df,id_col, cat_col,max_n=500,test_col="test", seed=1234){
  
  set.seed(1234)
  
  dat_split = split(x = df[,id_col],f = list(df[,cat_col],df[,test_col]))
  
  # browser()
  
  df_out = data.frame(matrix(nrow = 0, ncol = length(names(df))))
  names(df_out) = names(df)  
  
  for(split_num in 1:length(dat_split)){
    
    split_name = names(dat_split)[split_num]
    split_test = strsplit(x = split_name,split = "[.]")[[1]][2]
    
    # browser()
    
    if(split_test == 0){
      
      my_ids = dat_split[[split_num]]
      
      df_sub = df[df[,id_col] %in% my_ids,]
      # browser()
      
      if(length(my_ids) == 0){
        df_resub = df_sub
      } else if(length(my_ids) > max_n){
        df_resub = df_sub[df_sub[,id_col] %in% sample(my_ids,size=max_n,replace = FALSE),]
      } else if(length(my_ids) < max_n){
        # browser()
        df_resub = df_sub[match(sample(my_ids,size=max_n,replace = TRUE),df_sub[,id_col]),]
      } else if(length(my_ids) == max_n){
        df_resub = df_sub
      } 
      
      df_out = rbind(df_out,df_resub)    
      
    } else{
      my_ids = dat_split[[split_num]]
      df_sub = df[df[,id_col] %in% my_ids,]
      df_resub = df_sub
      df_out = rbind(df_out,df_resub)    
    }
  }
  # browser()
  
  df_out[,id_col] = make_unique(df_out[,id_col])
  
  return(df_out)
  
}

test_set = make_balanced_data(balanced_st_dat, id_col = "IMG_ID", cat_col = "source", max_n=1400)

table(test_set$source, test_set$test, test_set$surgery)

# write.csv(test_set, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/surgery_sda_sksub_stsub_20210301.csv",quote=FALSE,row.names = FALSE)

    ###
    ### Small set of Stanford data training
    ###

st_test = st_bl_dat
st_test$test = 0
st_test = st_test[complete.cases(st_test),]

st_test$st_machine = paste0(substr(st_test$IMG_ID,1,2),"_unknown") 
st_test$st_machine[st_test$machine %in% c("Stanford", "SU_uknown")] = "Stanford"
st_test$st_machine[substr(st_test$machine,nchar(st_test$machine)-1,nchar(st_test$machine)) =="ST"] = "SKSilentTrial"
st_test$st_machine[st_test$st_machine == "ST_unknown"] = "SKSilentTrial"
st_test$st_machine[st_test$machine %in% orig_machines] = "SKOrig"

st_test$id = unlist(lapply(strsplit(st_test$IMG_ID,"_"), function(x)x[1]))
st_test$us_num = unlist(lapply(strsplit(st_test$IMG_ID,"_"), function(x)x[3]))
st_test$age_wks = as.numeric(st_test$age_wks)


set.seed(1234)
st_train_sample = sample(unique(st_test$id[st_test$st_machine == "Stanford" & st_test$surgery == 0]), size = 20)
st_train_sample = sample(unique(st_test$id[st_test$st_machine == "Stanford" & st_test$surgery == 1]), size = 3)

st_test$st_machine[st_test$id %in% st_train_sample] = "Stanford_train"

set.seed(1234)
for(machine in unique(st_test$st_machine)){ ## BREAKING FOR ALL NON-STANFORD SOURCES
  
  uniq_ids = unique(st_test$id[st_test$st_machine == machine])
  
  if(machine == "Stanford"){
    test_ids = sample(uniq_ids, length(uniq_ids) - 5)
  } else{
    if(length(uniq_ids) < 10){
      test_ids = sample(uniq_ids, 1) 
    } else{
      test_ids = sample(uniq_ids, round(0.15*length(unique_ids)))
    }
  } 
  st_test$test[st_test$id %in% test_ids] = 1
}

head(st_test)

table(st_test$st_machine,st_test$surgery,st_test$test)
table(st_test$st_machine,st_test$surgery)
table(st_test$st_machine,st_test$test)

st_test$id[st_test$test == 1 & st_test$st_machine == "Stanford_train"]

st_test$source = st_test$st_machine

### imputing age at US
table(st_test$us_num, is.na(as.numeric(st_test$age_wks)))

plot(st_test$age_wks, st_test$us_num)

for(my_row in 1:nrow(st_test)){
  if(is.na(st_test$age_wks[my_row])){
    cat("\nTRUE 1\n")
    if(st_test$us_num[my_row] == 1){
      cat("\nTRUE 2\n")
      st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == 1]))
    } else{
      cat("\nTRUE 3\n")
      my_id = st_test$id[my_row]
      my_us_num = st_test$us_num[my_row]
      cat("\nmy_us_num\n")
      cat(my_us_num)
      df_sub = st_test[st_test$id == my_id,]
      
      if(any(as.numeric(df_sub$us_num) < my_us_num & !is.na(df_sub$age_wks))){
        cat("\nTRUE 4\n")
        comp_idx = which(df_sub$us_num == max(df_sub$us_num[df_sub$us_num < my_us_num]))[1]
        comp_df = df_sub[comp_idx,]
        comp_us_num =comp_df$us_num
        cat("\ncomp_us_num\n")
        cat(comp_us_num)       
        
        mean_diff = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num])) - mean(na.omit(st_test$age_wks[st_test$us_num == comp_us_num]))
        
        st_test$age_wks[my_row] = comp_df$age_wks + mean_diff
      } else{
        cat("\nTRUE 5\n")
        st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num]))        
      }
    }    
  }
}

# st_test[is.na(st_test$age_wks),]

for(my_row in 1:nrow(st_test)){
  if(is.na(st_test$age_wks[my_row])){
    cat("\nTRUE 5\n")
    st_test$age_wks[my_row] = mean(na.omit(st_test$age_wks[st_test$us_num == my_us_num]))        
  }
}


st_test = st_test[,c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")]
st_test$surgery = as.numeric(st_test$surgery)

### balance set

table(st_test$source, st_test$test)

max_n = max(c(table(st_test$source, st_test$test)))

# name_vec = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
balanced_st_dat = st_test[complete.cases(st_test),]
b_sample = sample(st_test$IMG_ID[st_test$source == "SKSilentTrial" & st_test$test == 0],max_n-1168)
s_sample = sample(st_test$IMG_ID[st_test$source == "Stanford" & st_test$test == 0],max_n-5, replace = TRUE)
balanced_st_dat = rbind(balanced_st_dat, st_test[st_test$IMG_ID %in% b_sample,])
balanced_st_dat = rbind(balanced_st_dat, st_test[match(s_sample, st_test$IMG_ID),])

table(balanced_st_dat$source,balanced_st_dat$test)

table(is.na(as.numeric(balanced_st_dat$age_wks)),balanced_st_dat$source)

balanced_st_dat$side = ifelse(balanced_st_dat$side == "Left", 0, 1)

# write.csv(balanced_st_dat, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/surgery_sda_sksub_stsub_20210226.csv",quote=FALSE,row.names = FALSE)


make_unique = function(ids_in){
  # browser()
  
  ids_out = c()
  
  iter = 0
  for(id in ids_in){
    if(id %in% unique(ids_in[duplicated(ids_in)])){
      id = paste0(id,"_",iter)
      iter=iter+1
    }
    ids_out = c(ids_out,id)
  }
  
  return(ids_out)
}

make_balanced_data = function(df,id_col, cat_col,max_n=500,test_col="test", seed=1234){
  
  set.seed(1234)
  
  df = df[complete.cases(df[,c(test_col,cat_col,id_col,"surgery")]),]
  
  dat_split = split(x = df[,id_col],f = list(df[,cat_col],df[,test_col]))
  
  # browser()
  
  df_out = data.frame(matrix(nrow = 0, ncol = length(names(df))))
  names(df_out) = names(df)  
  
  for(split_num in 1:length(dat_split)){
    
    split_name = names(dat_split)[split_num]
    split_test = strsplit(x = split_name,split = "[.]")[[1]][2]
    
    # browser()
    
    if(split_test == 0){
      
      my_ids = dat_split[[split_num]]
      
      df_sub = df[df[,id_col] %in% my_ids,]
      # browser()
      
      if(length(my_ids) == 0){
        df_resub = df_sub
      } else if(length(my_ids) > max_n){
        df_resub = df_sub[df_sub[,id_col] %in% sample(my_ids,size=max_n,replace = FALSE),]
      } else if(length(my_ids) < max_n){
        # browser()
        df_resub = df_sub[match(sample(my_ids,size=max_n,replace = TRUE),df_sub[,id_col]),]
      } else if(length(my_ids) == max_n){
        df_resub = df_sub
      } 
      
      df_out = rbind(df_out,df_resub)    
      
    } else{
      my_ids = dat_split[[split_num]]
      df_sub = df[df[,id_col] %in% my_ids,]
      df_resub = df_sub
      df_out = rbind(df_out,df_resub)    
    }
  }
  # browser()
  
  df_out[,id_col] = make_unique(df_out[,id_col])
  
  return(df_out)
  
}

test_set = make_balanced_data(balanced_st_dat, id_col = "IMG_ID", cat_col = "source", max_n=1400)

table(test_set$source, test_set$test, test_set$surgery)
table(test_set$source, test_set$test)

write.csv(test_set, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/surgery_sda_sksub_stsub_STTRAIN_20210301.csv",quote=FALSE,row.names = FALSE)

  ### double checking data




### SickKids test



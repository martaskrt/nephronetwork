
library(rjson)
library(splitstackshape)
library(RJSONIO)
library(V8)
set.seed(1234)


### Stanford data
raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune_train_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
stan_train_dat = test$get("dat")

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune_test_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
stan_test_dat = test$get("dat")


### UIowa data
raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune_train_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
ui_train_dat = test$get("dat")

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune_test_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
ui_test_dat = test$get("dat")


### SickKids
raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SickKidswST_filenames_20210411.json")
test = v8()
test$assign("dat", JS(raw_in))
sk_train_dat = test$get("dat")

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_newSTonly_filenames_20210411.json")
test = v8()
test$assign("dat", JS(raw_in))
sk_test_dat = test$get("dat")


    ###
    ### Creating datasheet
    ###

# str(sk_train_dat)

out_df = data.frame(matrix(nrow=0,ncol=8))
names(out_df) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")

  ### Add SickKids Training Data
for(id in names(sk_train_dat)){
  id_list = sk_train_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
    
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[!is.na(as.numeric(names(side_list)))]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "SickKids", 0, side_list[[i]]$Age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

head(out_df)
names(out_df) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")


    ## Add SickKids test data (Silent Trial)

st_df = read.csv("C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/SilentTrial_Datasheet.csv", header=TRUE,as.is=TRUE)

for(id in names(sk_test_dat)){
  id_list = sk_test_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
      
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[!is.na(as.numeric(names(side_list)))]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                id_num = substr(id,5,nchar(id))
                
                age_wks = st_df$age_at_US_wk[st_df$ID == id_num & st_df$US_num == i & st_df$view_side == side]
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "SickKids", 1, age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

tail(out_df)

sk_test_dat[["STID410"]]

    ## Add Stanford train data 
for(id in names(stan_train_dat)){
  id_list = stan_train_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
      
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[names(side_list) != "surgery"]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "Stan", 0, side_list[[i]]$Age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

tail(out_df)

    ## Add Stanford test data 
for(id in names(stan_test_dat)){
  id_list = stan_test_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
      
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[names(side_list) != "surgery"]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "Stan", 1, side_list[[i]]$Age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

tail(out_df)


    ## Add UIowa train data 
for(id in names(ui_train_dat)){
  id_list = ui_train_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
      
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[names(side_list) != "surgery"]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "UI", 0, side_list[[i]]$Age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

tail(out_df)

    ## Add UIowa test data 
for(id in names(ui_test_dat)){
  id_list = ui_test_dat[[id]]
  
  for(side in c("Left","Right")){
    
    side_list = id_list[[side]]
    
    if(length(side_list) > 0){
      
      side01 = ifelse(side == "Left", 1, 0)
      
      for(i in names(side_list)[names(side_list) != "surgery"]){
        
        if(length(side_list[[i]]$Age_wks) > 0 ){
          if(side_list$surgery != "NA"){
            if(all(c("sag","trv") %in% names(side_list[[i]]))){
              if(side_list[[i]]$Age_wks != "NA"){
                row_id = paste0(id, "_", side, "_", i)
                
                row_in = c(side_list[[i]]$sag, side_list[[i]]$trv, row_id, side_list$surgery, "UI", 1, side_list[[i]]$Age_wks, side01)
                names(row_in) = c("SAG_FILE","TRV_FILE","IMG_ID","surgery","source","test","age_wks","side")
                
                out_df = rbind(out_df, row_in)
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
}

tail(out_df)

    ## Add test data 
target_train_ids = sample(out_df$IMG_ID[out_df$test == 0],size = 200,replace = FALSE)
target_test_ids = sample(out_df$IMG_ID[out_df$test == 1],size = 200,replace = FALSE)

target_train = out_df[out_df$IMG_ID %in% target_train_ids,]
target_test = out_df[out_df$IMG_ID %in% target_test_ids,]

target_train$source = "TargetSample"
target_test$source = "TargetSample"

target_train$IMG_ID = paste0(target_train$IMG_ID, "_target")
target_test$IMG_ID = paste0(target_test$IMG_ID, "_target")

head(target_train)

out_df = rbind(out_df, target_train, target_test)

  ### make data match numbers: 
table(out_df$source)

make_ids_unique = function(id_vec){
  id_vec[duplicated(id_vec)] = paste0(id_vec[duplicated(id_vec)], "_", 1:length(id_vec[duplicated(id_vec)]))
  return(id_vec)
}

make_balanced_data_n = function(df, n_samples= 250, id_col = "IMG_ID", cat_col = "source", lab_col = "surgery", max_n=500,test_col="test"){
  
  dat_split = split(unlist(df[id_col]),
                    list(unlist(df[cat_col]),unlist(df[test_col])))
  
  df_out = data.frame(matrix(nrow = 0, ncol = length(names(df))))
  names(df_out) = names(df)  
  
  for(my_ids in dat_split){
    
    df_sub = df[unlist(df[id_col]) %in% my_ids,]
    
    # browser()
    
    if(unique(unlist(df_sub[test_col])) == 0){
      # dat_split2 = split(unlist(df_sub[id_col]), list(unlist(df_sub[lab_col])))
      ids_sub = my_ids
      
      # for(ids_list in dat_split2){
      #   ids_sub = c(ids_sub,sample(ids_list,n_samples,replace = TRUE))  
      # }
      ### SUBSET NUMBER OF IDS FROM **TEST=0 ONLY**
      
      if(length(ids_sub) == 0){
        df_resub = df_sub
      } else if(length(ids_sub) > max_n){
        df_resub = df_sub[unlist(df_sub[id_col]) %in% sample(ids_sub,size=max_n,replace = FALSE),]
      } else if(length(ids_sub) < max_n){
        df_resub = df_sub[match(sample(ids_sub,size=max_n,replace = TRUE),unlist(df_sub[id_col])),]
      } else if(length(ids_sub) == max_n){
        df_resub = df_sub
      } 

      # browser()
      
    } else{
      ids_sub = my_ids
      
      df_resub = df_sub
    }
    
    
    df_out = rbind(df_out,df_resub)    
    
  }
  
  return(df_out)
  
}


balanced_out_df = make_balanced_data_n(out_df)

balanced_out_df$IMG_ID = make_ids_unique(balanced_out_df$IMG_ID)

str(balanced_out_df)
table(balanced_out_df$source, balanced_out_df$test)

write.csv(balanced_out_df, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/surgery_sda_sk_st_stan_ui_subs_20210713.csv", quote = FALSE, row.names = FALSE)


small_balanced_out_df = make_balanced_data_n(out_df,max_n = 100)
small_balanced_out_df$IMG_ID = make_ids_unique(small_balanced_out_df$IMG_ID)

small_balanced_out_df[small_balanced_out_df$source == "UI",]
tail(small_balanced_out_df)
table(small_balanced_out_df$source, small_balanced_out_df$test)

write.csv(small_balanced_out_df, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/DA/datasheets/small_surgery_sda_sk_st_stan_ui_subs_20210713.csv", quote = FALSE, row.names = FALSE)


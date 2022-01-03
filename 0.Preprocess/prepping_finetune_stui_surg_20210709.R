
library(readxl)
library(reshape2)
library(rjson)
library(RJSONIO)


## read in and split Stanford data: 

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_StanOnly_filenames_20210612.json")
test = v8()
test$assign("dat", JS(raw_in))
stan_full_dat = test$get("dat")

stan_datasheetasheet = read_excel("C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/Datasheets/AI Hydronephrosis Data deidentified.xlsx")
head(stan_datasheetasheet)

names(stan_full_dat)
names(stan_full_dat[[1]])

stan_datasheet$l_surg = 0
stan_datasheet$l_surg[stan_datasheet$`Did you indicate surgery?` == "Yes" & stan_datasheet$`Which kidney was sx indicated for? (L/R)` == "Left"] = 1

stan_datasheet$r_surg = 0
stan_datasheet$r_surg[stan_datasheet$`Did you indicate surgery?` == "Yes" & stan_datasheet$`Which kidney was sx indicated for? (L/R)` == "Right"] = 1

stan_datasheet$`Did you indicate surgery? Binary` = ifelse(stan_datasheet$`Did you indicate surgery?` == "Yes", 1, 0)

stan_datasheet$`Did you indicate surgery at any time?` = stan_datasheet$`Did you indicate surgery? Binary`

for(i in 1:nrow(stan_datasheet)){
  stan_datasheet$`Did you indicate surgery at any time?`[i] = max(stan_datasheet$`Did you indicate surgery? Binary`[stan_datasheet$anon_mrn == stan_datasheet$anon_mrn[i]])
}
table(stan_datasheet$`Did you indicate surgery at any time?`)

stan_datasheet$l_surg_anytime = 0
stan_datasheet$l_surg_anytime[stan_datasheet$`Did you indicate surgery at any time?` == 1 & stan_datasheet$`side of hydronephrosis` == "Left"] = 1

stan_datasheet$r_surg_anytime = 0
stan_datasheet$r_surg_anytime[stan_datasheet$`Did you indicate surgery at any time?` == 1 & stan_datasheet$`side of hydronephrosis` == "Right"] = 1

table(stan_datasheet$r_surg_anytime)
table(stan_datasheet$l_surg_anytime)

unique(stan_datasheet$anon_mrn[stan_datasheet$r_surg_anytime == 1])
unique(stan_datasheet$anon_mrn[stan_datasheet$l_surg_anytime == 1])

unique(stan_datasheet$anon_mrn[stan_datasheet$r_surg_anytime == 0])
unique(stan_datasheet$anon_mrn[stan_datasheet$l_surg_anytime == 0])

  ## 30%
stan_surg_train = c("SU2bae8b2", "SU2bae8c5", "SU2bae883", "SU2bae885", "SU2bae893") ## will automatically include no surg
stan_nosurg_train = c("SU2bae87d", "SU2bae87e", "SU2bae87f", "SU2bae880", "SU2bae881", "SU2bae882",
                      "SU2bae883", "SU2bae884", "SU2bae885", "SU2bae886", "SU2bae887", "SU2bae888",
                      "SU2bae889", "SU2bae88a", "SU2bae88b", "SU2bae88c", "SU2bae88d", "SU2bae88e", 
                      "SU2bae87d", "SU2bae87e", "SU2bae87f", "SU2bae880", "SU2bae881", "SU2bae882", 
                      "SU2bae884", "SU2bae886", "SU2bae887", "SU2bae888", "SU2bae889", "SU2bae88a", 
                      "SU2bae88b", "SU2bae88c", "SU2bae88d", "SU2bae88e", "SU2bae88f", "SU2bae890", 
                      "SU2bae891", "SU2bae892", "SU2bae894", "SU2bae895", "SU2bae896", "SU2bae897", 
                      "SU2bae898", "SU2bae899", "SU2bae89a", "SU2bae89b", "SU2bae89c")


stan_train = unique(c(stan_surg_train, stan_nosurg_train))

length(stan_train)/length(unique(names(stan_full_dat))) ## 30% of patients

stan_train_dat = stan_full_dat[stan_train]
stan_test_dat = stan_full_dat[!(names(stan_full_dat) %in% stan_train)]

write(toJSON(stan_train_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune30%_train_20210711.json")
write(toJSON(stan_test_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune30%_test_20210711.json")

  ## 60%
stan_surg_train = c("SU2bae8b2", "SU2bae8c5", "SU2bae883", "SU2bae885", "SU2bae893") ## will automatically include no surg
stan_nosurg_train = c("SU2bae87d", "SU2bae87e", "SU2bae87f", "SU2bae880", "SU2bae881", "SU2bae882",
                      "SU2bae883", "SU2bae884", "SU2bae885", "SU2bae886", "SU2bae887", "SU2bae888",
                      "SU2bae889", "SU2bae88a", "SU2bae88b", "SU2bae88c", "SU2bae88d", "SU2bae88e", 
                      "SU2bae87d", "SU2bae87e", "SU2bae87f", "SU2bae880", "SU2bae881", "SU2bae882", 
                      "SU2bae884", "SU2bae886", "SU2bae887", "SU2bae888", "SU2bae889", "SU2bae88a", 
                      "SU2bae88b", "SU2bae88c", "SU2bae88d", "SU2bae88e", "SU2bae88f", "SU2bae890", 
                      "SU2bae891", "SU2bae892", "SU2bae894", "SU2bae895", "SU2bae896", "SU2bae897", 
                      "SU2bae898", "SU2bae899", "SU2bae89a", "SU2bae89b", "SU2bae89c", "SU2bae89d", 
                      "SU2bae89e", "SU2bae89f", "SU2bae8a0", "SU2bae8a1", "SU2bae8a2", "SU2bae8a3", 
                      "SU2bae8a4", "SU2bae8a5", "SU2bae8a6", "SU2bae8a7", "SU2bae8a8", "SU2bae8a9",
                      "SU2bae8aa", "SU2bae8ab", "SU2bae8ac", "SU2bae8ad", "SU2bae8ae", "SU2bae8af", 
                      "SU2bae8b0", "SU2bae8b1", "SU2bae8b3", "SU2bae8b4", "SU2bae8b5", "SU2bae8b6", 
                      "SU2bae8b7", "SU2bae8b8", "SU2bae8b9" ,"SU2bae8ba", "SU2bae8bb", "SU2bae8bc", 
                      "SU2bae8bd", "SU2bae8be", "SU2bae8bf")

stan_train = c(stan_surg_train, stan_nosurg_train)

length(unique(stan_train))/length(unique(names(stan_full_dat))) ## 30% of patients
length(names(stan_full_dat)[!(names(stan_full_dat) %in% stan_train)])

stan_train_dat = stan_full_dat[stan_train]
stan_test_dat = stan_full_dat[!(names(stan_full_dat) %in% stan_train)]

saveRDS(object = stan_train, 
        file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/Stan60%train_ids.rds")

write(toJSON(stan_train_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune60%_train_20210711.json")
write(toJSON(stan_test_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune60%_test_20210711.json")


  ## read in and split UIowa data: 

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIonly_filenames_20210612.json")
test = v8()
test$assign("dat", JS(raw_in))
ui_full_dat = test$get("dat")

ui_datasheet = read.csv("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UIowa_Datasheet2.csv",header=TRUE,as.is=TRUE)
head(ui_datasheet)

ui_datasheet$surgery = ifelse(substr(ui_datasheet$Name,1,1) == "O", 1, 0)

names(ui_full_dat)
names(ui_full_dat[[1]])

  ## 30%
ui_surg_train = c("O2","O6","O3","O4","O5","O6","O7","O8","O9","O10","O11","O12","O14","O15")
ui_nosurg_train = c("C3","C4","C6","C7","C9","C13","C10","C11","C20")

ui_train = c(ui_surg_train, ui_nosurg_train)

length(ui_train)/length(ui_full_dat) ## 30% training

length(names(ui_full_dat)[!(names(ui_full_dat) %in% ui_train)])

ui_train_dat = ui_full_dat[ui_train]
str(ui_train_dat)
ui_test_dat = ui_full_dat[!(names(ui_full_dat) %in% ui_train)]
str(ui_test_dat)

write(toJSON(ui_train_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune30%_train_20210711.json")
write(toJSON(ui_test_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune30%_test_20210711.json")

  ## 60%
ui_surg_train = c("O2","O6","O3","O4","O5","O6","O7","O8","O9","O10","O11","O12","O14","O15",
                  "O16", "O17", "O18", "O19", "O20", "O21", "O22", "O23", "O24", 
                  "O25", "O26", "O27", "O28", "O29", "O30", "O31", "O32", "O33",
                  "O34",  "O35",  "O36",  "O37")
ui_nosurg_train = c("C3","C4","C6","C7","C9","C13","C10","C11","C20","C22")

ui_train = c(ui_surg_train, ui_nosurg_train)

length(ui_train)/length(ui_full_dat) ## 60% training

ui_train_dat = ui_full_dat[ui_train]
str(ui_train_dat)
ui_test_dat = ui_full_dat[!(names(ui_full_dat) %in% ui_train)]
str(ui_test_dat)

saveRDS(object = ui_train, 
        file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/UI60%train_ids.rds")

write(toJSON(ui_train_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune60%_train_20210711.json")
write(toJSON(ui_test_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune60%_test_20210711.json")


    ##
    ##    COMBINED TRAINING
    ## 


raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SickKidswST_filenames_20210411.json")

library(V8)
test = v8()
test$assign("dat", JS(raw_in))
sk_list = test$get("dat")

length(sk_list)
names(sk_list)

# sk_list[["STID43"]]


combined_list = append(sk_list, stan_train_dat)


combined_list = append(combined_list, ui_train_dat)


names(combined_list)
combined_list[["O6"]][["Sex"]] = 2

lapply(combined_list, function(x)x$Sex)

for(name in names(combined_list)){
  for(side in c("Right","Left")){
    if(length(combined_list[[name]]$Sex) > 0){
      if(combined_list[[name]]$Sex == "M" | combined_list[[name]]$Sex == "F"){
        cat(name)
        cat("\n")
        cat(combined_list[[name]]$Sex)
        cat("\n")
        if(combined_list[[name]]$Sex == "M"){
          combined_list[[name]]$Sex = 1
        } else if(combined_list[[name]]$Sex == "F"){
          combined_list[[name]]$Sex = 2
        }
      }
    }
  }
}

write(toJSON(combined_list), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SKSTUI_60%train_20210722.json")


raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_newSTonly_filenames_20210411.json")

library(V8)
test = v8()
test$assign("dat", JS(raw_in))
sitri_list = test$get("dat")


c_list1 = append(sk_list, stan_train_dat)
write(toJSON(c_list1), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SKST_60%train_20210722.json")


c_list2 = append(sk_list, ui_train_dat)
write(toJSON(c_list2), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SKUI_60%train_20210722.json")

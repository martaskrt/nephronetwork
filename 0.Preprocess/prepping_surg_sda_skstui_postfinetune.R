
library(rjson)
library(splitstackshape)
set.seed(1234)


### Stanford data

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune_train_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
stan_train_dat = test$get("dat")

raw_in = readLines("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_Stan_finetune_test_20210711.json")
test = v8()
test$assign("dat", JS(raw_in))
stan_train_dat = test$get("dat")


### UIowa data

write(toJSON(ui_train_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune_train_20210711.json")
write(toJSON(ui_test_dat), "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_UIowa_finetune_test_20210711.json")


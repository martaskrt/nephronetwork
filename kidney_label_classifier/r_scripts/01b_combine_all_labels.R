library(readr)
library(dplyr)
library(purrr)
library(stringr)

# given all the labels combines all the kidney labels into a singular dataframe and 
# creates labels for all views (granular), views (view), and bladder + other (bladder)

setwd('~/Documents/Projects/nephro/')
label_dir <- '~/Documents/Projects/nephro/data/all_data/all_label_csv/'

# read in each individual csv and concatenate it 
print('Reading in each individual patients manifest')
infiles <- list.files(label_dir, full.names = TRUE)
# infiles
lbls <- infiles %>% 
  map_dfr(~ read_csv(.,
                    col_types = cols(
                      function_label = col_character(),
                     `image_acq_date:` = col_character(),
                     `image_acq_time:` = col_character(),
                     image_ids = col_character(),
                     image_manu = col_character(),
                     reflux_label = col_character(),
                     surgery_label = col_character(),
                     view_label = col_character())
                    )
          )

# 1096_1_4
# pid_ultrasound#_slice#

# extract the patient identifier and the high level view from the existing metadata

lbls <- lbls %>% 
  mutate(pid = str_split_fixed(image_ids, "_", n = 3)[, 1],
         view = str_split_fixed(view_label, pattern = '_', n = 2)[, 1]) %>%
  rename(image_label = view_label) %>% 
  mutate(numeric_image_label = as.factor(image_label) %>% as.numeric() - 1,
          numeric_view_label = as.factor(view) %>% as.numeric() - 1,
          numeric_bladder_label = ifelse(view == 'Bladder', 1, 0))

lbls

# examining the data
lbls %>% count(image_manu)

lbls %>% count(view)



print('Saving new manifests to ./data/all_kidney_manifest.csv')
write_csv(lbls, 'data/all_kidney_manifest.csv')



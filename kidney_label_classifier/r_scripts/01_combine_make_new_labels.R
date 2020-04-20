library(readr)
library(dplyr)
library(purrr)

setwd('~/Documents/Projects/nephro/')
label_dir <- '~/Documents/Projects/nephro/data/labels/'

# read in each individual csv and concatenate it 
print('Reading in each individual patients manifest')
infiles <- list.files(label_dir, full.names = TRUE)
# infiles
lbls <- infiles %>% 
  map_dfr(~ read_csv(.))

# 1096_1_4
# pid_ultrasound#_slice#

# extract the patient identifier and the high level view from the existing metadata
 
lbls <- lbls %>% 
  mutate(pid = str_split_fixed(image_ids, "_", n = 3)[, 1],
         view = str_split_fixed(image_label, pattern = '_', n = 2)[, 1])

# examining the data
lbls %>% count(image_manu)
lbls %>% count(image_label)
lbls %>% count(view)

set.seed(1)
# randomly sample 20% of the patient ids for the validation set 
val_pids <- lbls  %>% distinct(pid) %>% sample_frac(size = 0.2) 
# the remaining 80% is the training set
train_pids <- lbls %>% distinct(pid) %>% anti_join(val_pids)
# sanity check
# val_pids$pid %in% train_pids$pid

# convert the labels to numeric
new_lbls <- lbls %>% mutate(set = ifelse(pid %in% val_pids$pid, 'valid', 'train'),
                            numeric_image_label = as.factor(image_label) %>% as.numeric() - 1,
                            numeric_view_label = as.factor(view) %>% as.numeric() - 1,
                            numeric_bladder_label = ifelse(view == 'Bladder', 1, 0))


# sanity checks that validation IDs are not in training IDs
new_lbls$pid[new_lbls$set == 'valid'] %>% unique() %in% (val_pids$pid)
new_lbls$pid[new_lbls$set == 'valid'] %>% unique() %in% (train_pids$pid)

# wide table with set counts for view labels 
new_lbls %>%  count(set,  view)  %>% 
  spread(set, n) %>%
  group_by(view) %>%
  mutate(val_prop = valid/train, class_prop = (train+valid)/sum(.$train+.$valid)) %>% 
  ungroup() %>% mutate(mean = mean(val_prop))

# wide table with set counts for granular view labels 
new_lbls %>%  count(set, image_label)  %>% 
  spread(set, n) %>%
  group_by(image_label) %>%
  mutate(val_prop = valid/train, class_prop = (train+valid)/sum(.$train+.$valid)) %>% 
  ungroup() %>% mutate(mean = mean(val_prop))

print(paste('Training Observations:', nrow(new_lbls[new_lbls$set == 'train',])))
print(paste('Validation Observations:',  nrow(new_lbls[new_lbls$set == 'valid',])))

print('Saving new manifests to ./data/kidney_manifest.csv')
write_csv(new_lbls, 'data/kidney_manifest.csv')
# write_csv(train_lbls, 'data/train_manifest.csv')

new_lbls %>% distinct(image_label, numeric_image_label) %>% arrange(numeric_image_label)
new_lbls %>% distinct(view, numeric_view_label)


new_lbls %>% count(numeric_image_label)


# ---- create a sample dataset ----
set.seed(1)
n = 2000 # total size of the sample dataset

sample_nephro <- bind_rows(new_lbls %>% 
                             filter(set == 'train') %>% 
                             sample_n(n * 0.8),
                          new_lbls %>% 
                            filter(set == 'valid') %>%
                            sample_n(n * 0.2)) %>%  
  # create a column for the image names
  mutate( fn = paste0(image_ids, '.jpg'))
# sanity check that no IDs are across the two
sample_nephro %>% distinct(pid, set) %>% count(pid, sort = TRUE) %>% filter(n != 1)

# wide table with set counts for granular view labels 
sample_nephro %>% count(set, image_label)  %>% 
  spread(set, n) %>%
  group_by(image_label) %>%
  mutate(val_prop = valid/train, class_prop = (train+valid)/sum(.$train+.$valid)) %>% 
  ungroup() %>% mutate(mean = mean(val_prop))

# wide table with set counts for view labels 
sample_nephro %>% count(set, view)  %>% 
  spread(set, n) %>%
  group_by(view) %>%
  mutate(val_prop = valid/train, class_prop = (train+valid)/sum(.$train+.$valid)) %>% 
  ungroup() %>% mutate(mean = mean(val_prop))

nrow(sample_nephro)

write_csv(sample_nephro, 'data/sample_nephro.csv')
write_csv(sample_nephro %>% select(fn), 'data/sample_nephro_ids.csv')

# cd ~/Documents/Projects/nephro
# cp $(cat sample_nephro_ids.csv) ~/Documents/Projects/nephro/nephro_test/data/imgs/

# mn <- read_csv('~/Documents/Projects/nephro/data/kidney_manifest.csv')
# mn %>%
#   count(set, numeric_bladder_label)  %>% 
#   spread(set, n) %>%
#   group_by(numeric_bladder_label) %>%
#   mutate(val_prop = valid/train, class_prop = (train+valid)/sum(.$train+.$valid)) %>% 
#   ungroup() %>% mutate(mean = mean(val_prop))

library(ggplot2)
library(readr)
library(dplyr)



# ---- thresholds from the best model, calculated using sklearn.metrics ----
thresh <- read_csv('~/Documents/Projects/nephro/output/all_custom_threhsolds-1.csv')
# fpr = 1 - sp
# sp = 1 - fpr
mx <- thresh %>% mutate(youden = tpr + (1 - fpr)) %>%
  filter(youden == max(youden))
mx
# results in an FPR of 7.5 and TPR = 0.946
# this means 7.5% of images will be falsely bladder
# alternatively, this means 94.6% of the images which are bladders, will actually be classified as bladders

# visualize the AURC 

thresh %>% 
  mutate(youden = tpr + (1 - fpr), 
         max = ifelse(youden == max(youden), 1, 0),
         max = as.factor(max)) %>%
  ggplot(aes(x = fpr, y = tpr)) + 
  geom_line() + 
  geom_point(aes(colour = max, size = max), alpha = 1, show.legend  = FALSE) + 
  scale_size_manual(values = c(0.25, 3)) + 
  geom_label(aes(x = mx$fpr, y = mx$tpr, 
                 label = paste0('Threshold = ',
                                round(mx$threshold, 3)),
                 hjust = -0.05, vjust = -0.5)) + 
  geom_abline(aes(intercept = 0, slope = 1), linetype = 'dotted')  +
  scale_colour_brewer(palette = 'Set1') + 
  hrbrthemes::theme_ipsum_rc(base_size = 16) + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(x = '1 - Specificity (FPR)', y = 'Sensitivity (TPR)') +
  ggtitle('ROC - Bladder vs Other (Validation Set, n = 3858)', 'AUROC = 0.98')

# ---- the predictions ----
# what is the confusion matrix after applying the threshold?

best_preds <- read_csv('~/Documents/Projects/nephro/output/bladder_custom_no_wts_valid.csv')

nrow(best_preds) == 3858 # sanity check

best_preds <- best_preds %>% mutate(pred_class = ifelse(preds >= mx$threshold, 1, 0),
                                    labels = factor(ifelse(labels == 0, 'other', 'bladder'), 
                                                    levels = c('other', 'bladder')))

best_preds %>%
  count(pred_class, labels) %>% 
  mutate(pred_class = as.factor(pred_class),
         labels = factor(labels, levels = c('bladder', 'other'))) %>%
  ggplot(aes(x = pred_class, y = labels)) + 
  geom_tile(aes(fill = n), show.legend = FALSE) + 
  geom_label(aes(label = n), size = 10) + 
  scale_fill_viridis_c() +
  coord_equal() +
  theme_minimal() +
  theme(axis.text = element_text(size = 14))

# caret::confusionMatrix(as.factor(best_preds$labels), as.factor(best_preds$pred_class), positive = '1')
# cm = as.matrix(table(best_preds$preds >=mx$threshold, best_preds$labels))
cm = as.matrix(table(best_preds$pred_class, best_preds$labels))

tp = cm[2,2]
fp = cm[2,1]
fn = cm[1,2]
tn = cm[1,1]

ppv = tp/(tp + fp)
sn = tp/(tp + fn)
sp = tn/(tn + fp)


ppv;sn;( 1 - sp)
# matches up, sn = 94.6 and 1 - sp = 7.45%

# 
# best_roc <- pROC::roc(response = best_preds$labels, predictor = best_preds$preds)
# pROC::ggroc(best_roc)
# thresh_tbl <- tibble(thresholds = best_roc$thresholds, sp = best_roc$specificities, sn = best_roc$sensitivities)
# thresh_tbl %>% 
#   filter(thresholds > 0, thresholds < 1000) %>%
#   mutate(youden = sn + (1 - sp)) %>%
#   filter(youden == max(youden))
# 
# thresh_tbl %>% 
#   filter(thresholds > 0.01, thresholds < 100) %>%
#    mutate(youden = sn + (1 - sp),
#           max = ifelse(youden == max(youden), 1, 0),
#           max = as.factor(max)) %>%
#   ggplot(aes(x =1 - sp, y = sn)) + 
#   geom_line() + 
#   geom_point(aes(colour = max, size = max), alpha = 1, show.legend  = FALSE) +
#   scale_size_manual(values = c(0.25, 3)) +
#   # geom_label(aes(x = mx$fpr, y = mx$tpr, 
#   #                label = paste0('Threshold = ',
#   #                               round(mx$threshold, 3)),
#   #                hjust = -0.05, vjust = -0.5)) + 
#   geom_abline(aes(intercept = 0, slope = 1), linetype = 'dotted')  +
#   scale_colour_brewer(palette = 'Set1') + 
#   hrbrthemes::theme_ipsum_rc(base_size = 16) + 
#   theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
#   labs(x = '1 - Specificity (FPR)', y = 'Sensitivity (TPR)') +
#   ggtitle('ROC - Bladder vs Other (Validation Set, n = 3858)', 'AUROC = 0.98')


# apply threshold to probabilities ----

probs <- read_csv('~/Documents/Projects/nephro/output/bladder_probs.csv')
mn <- read_csv('~/Documents/Projects/nephro/data/kidney_manifest.csv') %>%
  mutate(image_ids = paste0(image_ids, '.jpg')) #%>% 
  # filter(numeric_bladder_label == 1)

nrow(probs) # 72459
nrow(mn) # 23523

# get the subset of images that are completely new (ie. unlabelled)
probs_new <- probs %>% anti_join(mn, by = c('img_dir' = 'image_ids'))
# threshold them
probs_new <-  mutate(probs_new, pred_label = ifelse(preds >= mx$threshold, 1, 0))

probs_new %>% count(pred_label) %>% mutate(perc = prop.table(n) * 100) # 15%

# get the images that are known to be bladders
known_bladder <- mn %>% filter(numeric_bladder_label == 1)
nrow(known_bladder)

# join the two dataframes together so we have only bladder imgs (for known labels) and predicted
all <- probs_new %>% filter(pred_label == 1) %>% 
  bind_rows(known_bladder %>% select(img_dir = image_ids)) %>%
  select(img_dir)
nrow(all)
all %>%
  write_csv('~/Documents/Projects/nephro/output/predicted_bladder_imgs.csv')


# not log transformed
# 
# probs %>% 
#   ggplot(aes(x = (preds))) + 
#   geom_vline(aes(xintercept = (0.0990))) + 
#   geom_histogram(aes(fill = ..count..)) + 
#   scale_fill_viridis_c()
# 
# # log transformed
# probs %>% 
#   ggplot(aes(x = log(preds))) + 
#   geom_vline(aes(xintercept = log(0.0990))) + 
#   geom_histogram(aes(fill = ..count..)) + 
#   scale_fill_viridis_c() +
#   theme_minimal() 
# 
# probs <- mutate(probs, pred_label = ifelse(preds >= mx$threshold, 1, 0))
# 
# probs %>% count(pred_label) %>% mutate(prop = n/sum(n))
# 
# 
# probs %>% 
#   filter(pred_label == 1) %>%
#   select(img_dir) %>%
#   write_csv('~/Documents/Projects/nephro/output/predicted_bladder_imgs.csv')
# probs

# the proportion of positives are 18.7 and 16% in the training and validation set, respectively
# 12.4%.

# what is worse - sifting through and 
# having to remove a lot of false positives (incorrectly classified bladders),
# lower FPR -> less removal of false positives 
# or actually missing out on a lot of positives (ie. having a lot of false negatives)



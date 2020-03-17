library(readr)
library(dplyr)
library(ggplot2)
library(purrr)
library(stringr)

##
wd = '~/Documents/Projects/nephro/output' # or whever these files are saved
infiles <- list.files(wd, pattern = 'softmax')

infiles_abs <- file.path(wd, infiles)

probs <- infiles_abs %>% 
  map_dfr(~ read_csv(.) %>%
            mutate(labels = as.factor(labels),
                   class = as.factor(class),
                   fn = str_split_fixed(.x, "/", n =6 )[,6]))
probs


# vectors for renaming classes
# pulled from the manifest 
# image_label      numeric_image_label
# <chr>                          <dbl>
#   1 Bladder                            0
# 2 Other                              1
# 3 Saggital_Left                      2
# 4 Saggital_Right                     3
# 5 Transverse_Left                    4
# 6 Transverse_Right                   5
granular_labels <- c('Bladder', 'Other', 'Sagittal_Left', 
                     'Sagittal_Right', 'Transverse_Left', 'Transverse_Right')
names(granular_labels) <- c(0:5)

bladder_labels <- c('Other', 'Bladder')
names(bladder_labels)  <- c(0:1)

# ---- granular -----

probs_g <- probs %>%
  filter(task == 'granular') %>% 
  mutate(correct = as.factor(ifelse(labels == class, 1, 0))) 


png(filename = file.path(wd, 'granular_probs_fig.png'), width = 900, height = 600 )
probs_g %>%   
  filter(wts == 'no_wts') %>%
  # filter(task == 'granular', wts == 'no_wts', mod == 'custom') %>%
  mutate(labels = ifelse(labels %in% names(granular_labels), granular_labels[labels], NA),
         class = ifelse(class %in% names(granular_labels), granular_labels[class], NA)) %>%
  # distinct(pod)
  # mutate(probs = log(probs)) %>% 
  ggplot(aes(x = class, y = probs, fill = correct)) + 
  geom_violin(trim = FALSE, 
              alpha = 0.8,
              scale = 'width',
              position = position_dodge(width = 1)) +
  # geom_jitter(aes(colour = correct, group = correct), alpha = 0.8, position = position_jitterdodge(), size = 1) +
  geom_boxplot(alpha = 0.8, width = 0.25, outlier.shape = NA, position = position_dodge(width = 1)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  hrbrthemes::theme_ipsum_rc(base_size = 20) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 16)) + 
  facet_wrap(~ fn, scales = 'free_x') +
  labs(y = 'Probability of Image being in given class', x = '') +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25))
dev.off()
  
probs_b <- probs %>% 
  filter(task == 'bladder') %>% 
  mutate(correct = as.factor(ifelse(labels == class, 1, 0))) 

png(filename = file.path(wd, 'bladder_probs_fig.png'), width = 840, height = 600 )
probs_b %>%   
  # filter(task == 'bladder', wts == 'no_wts', mod == 'custom') %>%
  mutate(labels = ifelse(labels %in% names(bladder_labels), bladder_labels[labels], NA),
         class = ifelse(class %in% names(bladder_labels), bladder_labels[class], NA)) %>%
  # distinct(pod)
  mutate(correct = as.factor(ifelse(labels == class, 1, 0))) %>% 
  # mutate(probs = log(probs)) %>% 
  ggplot(aes(x = class, y = probs, fill = correct)) + 
  geom_violin(trim = FALSE, 
              alpha = 0.8,
              scale = 'width',
              position = position_dodge(width = 1)) +
  # geom_jitter(aes(colour = correct, group = correct), alpha = 0.8, position = position_jitterdodge(), size = 1) +
  geom_boxplot(alpha = 0.8, width = 0.25, outlier.shape = NA, position = position_dodge(width = 1)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  hrbrthemes::theme_ipsum_rc(base_size = 20) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 16)) + 
  facet_wrap(~ fn, scales = 'free_x') +
  labs(y = 'Probability of Image being in given class', x = '') +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25))
dev.off()

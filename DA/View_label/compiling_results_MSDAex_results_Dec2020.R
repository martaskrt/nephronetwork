



csv_dir = "C:/Users/lauren erdman/Desktop/kidney_img/View_Labeling/OutputNov2020/csv/"

setwd(csv_dir)

## Experiment 1
ex_1_domain = c("ACUSON", "ATL","Philips_Medical_Systems","TOSHIBA_MEC")
ex_1_msda = c("ACUSON_2_test.txt", "ATL_0_test.txt","Philips_Medical_Systems_2_test.txt","TOSHIBA_MEC_0_test.txt")
ex_1_nomsda = c("NoMSDA_ACUSON_0_test.txt","NoMSDA_ATL_0_test.txt","NoMSDA_Philips_Medical_Systems_1_test.txt","NoMSDA_TOSHIBA_MEC_0_test.txt")
ex_1_domain_ss = c(57, 140, 500, 500)

## Experiment 2
ex_2_domain = c("Stanford")
ex_2_msda = c("Stanford_0_test_LR0.0001.txt")
ex_2_nomsda = c("NoMSDA_Stanford_0_test_LR0.0001.txt")
ex_2_domain_ss = c(108)

## Experiment 3 
ex_3_domain = c("GE Healthcare_LOGIQE9", "ACUSON_SEQUOIA", "SIEMENS_S3000")
ex_3_msda = c("GE Healthcare_LOGIQE9_0_test.txt", "ACUSON_SEQUOIA_0_test.txt", "SIEMENS_S3000_0_test.txt")
ex_3_nomsda = c("NoMSDA_GE Healthcare_LOGIQE9_0_test.txt", "NoMSDA_ACUSON_SEQUOIA_1_test.txt", "NoMSDA_SIEMENS_S3000_0_test.txt")
ex_3_domain_ss = c(41, 32, 108) ## check these numbers! 

## Experiment 4 
ex_4_domain = c("labeled73","labeled6")
ex_4_msda = c("SIEMENS_S3000_0_test.txt", "3examples_SIEMENS_S3000_1_test.txt")
ex_4_nomsda = c("NoMSDA_SIEMENS_S3000_0_test.txt", "3examples_NoMSDA_SIEMENS_S3000_1_test.txt")
ex_4_domain_ss = c(108, 108) ## check these numbers! 

## Experiment 5
ex_5_groups = c("labled6")
ex_5_results = c("SIEMENS_S3000_0_test.txt", "3ex_NoMSDA_SIEMENS_S3000_4_test.txt")
ex_5_ss = c(108, 108, 108) ## check these numbers! 


## FUNCTIONS

get_cis = function(prop, n, n_resamples=1000){
  
  n_corr = round(prop*n)
  n_incorr = round((1-prop)*n)
  
  est_vec = c(rep(0,n_incorr),rep(1,n_corr))
  
  acc_vec = c()
  for(i in 1:n_resamples){
    my_acc_samp = mean(sample(est_vec, size = n, replace = TRUE))
    acc_vec = c(acc_vec, my_acc_samp)
  }
  
  acc_vec = sort(acc_vec)
  out_vec = quantile(acc_vec, c(0.025, 0.975)) 
  names(out_vec) = c("lcl","ucl")
  
  return(out_vec)
}

make_results_graph_df = function(domains, msda_files, nomsda_files, domain_ss){

  names(domain_ss) = domains
  names(msda_files) = domains
  names(nomsda_files) = domains
  
  col_names = c("Accuracy","Domain","M3SDA","Epoch","LCL","UCL")
  in_dat = data.frame(matrix(nrow = 0, ncol = length(col_names)))
  
  for(domain in domains){
    browser()
    my_file = read.table(msda_files[domain], header=FALSE)
    names(my_file) = "Accuracy"
    my_file$Domain = domain
    my_file$M3SDA = 1
    my_file$Epoch = 1:nrow(my_file)
    cis = sapply(my_file$Accuracy,function(x){get_cis(prop = x, n = domain_ss[domain])})
    my_file$LCL = cis["lcl",]
    my_file$UCL = cis["ucl",]
    
    in_dat = rbind(in_dat, my_file)
    my_file = read.table(nomsda_files[domain], header=FALSE)
    names(my_file) = "Accuracy"
    my_file$Domain = domain
    my_file$M3SDA = 0
    my_file$Epoch = 1:nrow(my_file)
    cis = sapply(my_file$Accuracy,function(x){get_cis(prop = x, n = domain_ss[domain])})
    my_file$LCL = cis["lcl",]
    my_file$UCL = cis["ucl",]
    
    in_dat = rbind(in_dat, my_file)
  }
  in_dat$M3SDA  = factor(in_dat$M3SDA, levels = c(0,1), labels = c("NoM3SDA","M3SDA"))
  return(in_dat)
}

make_results_graph_2 = function(group_names, result_files, domain_ss){
  
  names(domain_ss) = group_names
  names(result_files) = group_names

  col_names = c("Accuracy","Group","M3SDA","Epoch","LCL","UCL")
  in_dat = data.frame(matrix(nrow = 0, ncol = length(col_names)))
  
  for(group in group_names){
    # browser()
    my_file = read.table(msda_files[group], header=FALSE)
    names(my_file) = "Accuracy"
    my_file$Group = group
    my_file$Epoch = 1:nrow(my_file)
    cis = sapply(my_file$Accuracy,function(x){get_cis(prop = x, n = domain_ss[group])})
    my_file$LCL = cis["lcl",]
    my_file$UCL = cis["ucl",]
    
    in_dat = rbind(in_dat, my_file)
  }
  in_dat$Group = factor(in_dat$Group)
  return(in_dat)
}

## GRAPHS

theme_set(
  theme_bw(base_size = 20)
)

## EXERCISE 1
ex1 = make_results_graph_df(domains = ex_1_domain, msda_files = ex_1_msda, nomsda_files = ex_1_nomsda, domain_ss = ex_1_domain_ss)

head(ex1)

ggplot(ex1, aes(x = Epoch, y = Accuracy, col = M3SDA, group = M3SDA)) + 
  geom_errorbar(aes(ymin = LCL, ymax = UCL), width=0.1, position = position_dodge(0.1), size = 1) + 
  geom_line(position = position_dodge(0.1), size = 1) + 
  geom_point(position = position_dodge(0.1), size = 2) + 
  facet_grid(~Domain)

## EXERCISE 2
ex2 = make_results_graph_df(domains = ex_2_domain, msda_files = ex_2_msda, nomsda_files = ex_2_nomsda, domain_ss = ex_2_domain_ss)

head(ex2)

ggplot(ex2, aes(x = Epoch, y = Accuracy, col = M3SDA, group = M3SDA)) + 
  geom_errorbar(aes(ymin = LCL, ymax = UCL), width=0.1, position = position_dodge(0.1), size = 1) + 
  geom_line(position = position_dodge(0.1), size = 1) + 
  geom_point(position = position_dodge(0.1), size = 2)


## EXERCISE 3
ex3 = make_results_graph_df(domains = ex_3_domain, msda_files = ex_3_msda, nomsda_files = ex_3_nomsda, domain_ss = ex_3_domain_ss)

head(ex3)

ggplot(ex3, aes(x = Epoch, y = Accuracy, col = M3SDA, group = M3SDA)) + 
  geom_errorbar(aes(ymin = LCL, ymax = UCL), width=0.1, position = position_dodge(0.1), size = 1) + 
  geom_line(position = position_dodge(0.1), size = 1) + 
  geom_point(position = position_dodge(0.1), size = 2) + 
  facet_grid(~Domain)


## EXERCISE 4
ex4 = make_results_graph_df(domains = ex_4_domain, msda_files = ex_4_msda, nomsda_files = ex_4_nomsda, domain_ss = ex_4_domain_ss)

head(ex4)

ggplot(ex4, aes(x = Epoch, y = Accuracy, col = M3SDA, group = M3SDA)) + 
  geom_errorbar(aes(ymin = LCL, ymax = UCL), width=0.1, position = position_dodge(0.1), size = 1) + 
  geom_line(position = position_dodge(0.1), size = 1) + 
  geom_point(position = position_dodge(0.1), size = 2) + 
  facet_grid(~Domain)


## EXERCISE 5
ex5 = make_results_graph_2(group_names = )

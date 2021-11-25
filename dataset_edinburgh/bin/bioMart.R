library(biomaRt)
library(dplyr)
library(Biobase)
library(groHMM)

# running these to resolve ssl certificate issues with ensembl
new_config <- httr::config(ssl_verifypeer = FALSE)
httr::set_config(new_config, override = FALSE)

# get the vector of gene codes looking for from existing analysis
rawdata <- read.csv('~/Documents/GitHub/sepsis_ml_omics_msc/gene_codes_df.csv',header = T)
ilmn_codes <- rawdata$Probe_Id

# select the biomaRt database that you want to connect to - basically always "ensembl". creates object
ensembl <- useMart("ensembl")

# select a dataset within the BioMart database selected with the usemart function. Here selecting human genes out of >200 options.
ensembl <- useDataset("hsapiens_gene_ensembl", mart=ensembl)

# query the selected database - this is the main query call. # not used: values = featureNames(exp.norm).
annotationDF <- getBM(attributes = c('illumina_humanht_12_v3', 'ensembl_gene_id', 'external_gene_name', 'description', 'gene_biotype'), filters = 'illumina_humanht_12_v3', values = ilmn_codes, mart = ensembl)

# write the annotations to csv file
write.csv(annotationDF,"~/Documents/GitHub/sepsis_ml_omics_msc/dataset_edinburgh/illuminaht12v3_ensembl_mapping.csv", row.names = FALSE)
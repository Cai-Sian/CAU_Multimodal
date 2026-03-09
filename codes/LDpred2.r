#!/bin/Rscript

# LDpred (Quantitative traits)

rm(list = ls())

# Load packages bigsnpr and bigstatsr
library(bigsnpr)
library(dplyr)
library(ggplot2)

NCORES = 96
# Read from bed/bim/fam, it generates .bk and .rds files.
snp_readBed("../genetic_analysis_inputs/TWB_genotype_CAU_val.bed")


for (Pheno_var in c("PS","MD","ED","TAMAX","PI","RI","diameter","IMT")){    
print(Pheno_var)
obj.bigSNP = snp_attach("../genetic_analysis_inputs/TWB_genotype_CAU_val.rds")
str(obj.bigSNP, max.level = 2, strict.width = "cut")
    
# set phenotype
sample_order = obj.bigSNP$fam


G = obj.bigSNP$genotypes
str(obj.bigSNP$map)
genotype = snp_fastImputeSimple(G)
obj.bigSNP$genotypes = genotype
G = obj.bigSNP$genotypes

CHR = obj.bigSNP$map$chromosome
POS = obj.bigSNP$map$physical.pos

# Read summary statistics
print(sprintf("../Results/GWAS_result/CAU_average_%s_training.%s.glm.linear",Pheno_var,Pheno_var))
sumstats = bigreadr::fread2(sprintf("../Results/GWAS_result/CAU_average_%s_training.%s.glm.linear",Pheno_var,Pheno_var))

str(sumstats)
colnames(sumstats)
sumstats = sumstats[sumstats['P']<0.05,]
str(sumstats)
sumstats = subset(sumstats, select = -c(A1, TEST,T_STAT,ERRCODE))

## continuous
colnames(sumstats) =c('chr','pos','rsID','a0','a1','n_eff','beta','beta_se','p')

####
map = dplyr::transmute(obj.bigSNP$map,
                        chr = chromosome, pos = physical.pos,rsID = marker.ID,
                        a0 = allele2, a1 = allele1)


df_beta = snp_match(sumstats,map)



str(df_beta)


POS2 = snp_asGeneticPos(CHR, POS, dir = "../genetic_analysis_inputs/PRS/LDpred2/tmp-data", ncores = NCORES)

tmp = tempfile(tmpdir = "../genetic_analysis_inputs/PRS/LDpred2/tmp-data")
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)


for (chr in 1:22) {
  
  # print(chr)
  
  ind.chr = which(df_beta$chr == chr)
  ind.chr2 = df_beta$`_NUM_ID_`[ind.chr]
  
  corr0 = snp_cor(G, ind.col = ind.chr2, size = 3 / 1000,
                   infos.pos = POS2[ind.chr2], ncores = NCORES)
  
  if (chr == 1) {
    ld = Matrix::colSums(corr0^2)
    corr = as_SFBM(corr0, tmp, compact = TRUE)
  } else {
    ld = c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

saveRDS(corr, file = paste("../genetic_analysis_inputs/PRS/LDpred2/tmp-data/corr_CAU_average_",Pheno_var,"_val.rds",sep=""))
saveRDS(ld, file = paste("../genetic_analysis_inputs/PRS/LDpred2/tmp-data/ld_CAU_average_",Pheno_var,"_val.rds",sep=""))


(ldsc = with(df_beta, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2,sample_size = n_eff, blocks = NULL)))
ldsc_h2_est = ldsc[["h2"]]


# auto

if (Pheno_var == 'diameter'){
    coef_shrink = 0.8  ## to let "diameter" coverage
} else{
    coef_shrink = 0.95}

set.seed(1) 
multi_auto = snp_ldpred2_auto(
  corr, df_beta, h2_init = ldsc_h2_est,
  vec_p_init = seq_log(1e-4, 0.2, length.out = 30), ncores = NCORES,
  allow_jump_sign = FALSE, shrink_corr = coef_shrink)
str(multi_auto, max.level = 1)



library(ggplot2)


auto = multi_auto[[1]] 

png(sprintf("../Results/PRS/LDpred2_plot_%s.png",Pheno_var))

plot_grid(
  qplot(y = auto$path_p_est) + 
    theme_bigstatsr() + 
    geom_hline(yintercept = auto$p_est, col = "blue") +
    scale_y_log10() +
    labs(y = "p"),
  qplot(y = auto$path_h2_est) + 
    theme_bigstatsr() + 
    geom_hline(yintercept = auto$h2_est, col = "blue") +
    labs(y = "h2"),
  ncol = 1, align = "hv"
)

dev.off()



(range = sapply(multi_auto, function(auto) diff(range(auto$corr_est))))
(keep = which(range > (0.95 * quantile(range, 0.95, na.rm = TRUE))))
beta_auto = rowMeans(sapply(multi_auto[keep], function(auto) auto$beta_est))


beta_final = cbind(df_beta,beta_auto)

write.table(beta_final,sprintf("../Results/PRS/CAU_average_%s_LDpred2_wieghts.txt",Pheno_var), quote=F,row.names=F,col.names=T,sep="\t")

mydir = "../genetic_analysis_inputs/PRS/LDpred2/tmp-data"
files_to_delete = dir(path=mydir ,pattern="*.sbk")
file.remove(file.path(mydir, files_to_delete)) 
                                 }
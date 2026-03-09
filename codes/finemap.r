#!/bin/Rscript

# finemapping

library(data.table)
library(susieR)

assoc <- sapply(c("PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"),function(x) fread(sprintf("../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_%s_window1000000.txt",x)),simplify=F)

#### find causal variants in each group
for (pheno in (c("PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"))){             
    print(pheno)
    assoc_1 = assoc[[pheno]]
    assoc_clean_1 = assoc_1[group!=""]
    print(unique(assoc_clean_1$group))

    finemap_analysis = data.frame()
    for (i in (unique(assoc_clean_1$group))){
        print(i)
        R = read.table(sprintf("../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_%s_window1000000_%s_r.ld",pheno,i), header = FALSE, sep = "\t", stringsAsFactors = FALSE)
        R = as.matrix(R)
        
        assoc_2 = assoc_1[assoc_1$group == i,]

        fitted_rss2 = susie_rss(bhat = assoc_2$beta, shat = assoc_2$SE, R = R, n = unique(assoc_2$N),var_y = (unique(assoc_2$Var_Y)), L = 10,estimate_residual_variance = TRUE)

        result = summary(fitted_rss2)$cs
        pip = fitted_rss2$pip
        print(result)
        if (length(result) == 0){
            print(sprintf('No snp found after finemapped in %s of %s',i,pheno))}
        else {
        for (z in unique(summary(fitted_rss2)$cs$cs)){
            print(z)
            result2 = result[result['cs'] == z,]
            print(result2)
        
            split_values <- strsplit(result2$variable, ",")[[1]]
        
            # Convert the character vector to a numeric vector
            numeric_values <- as.numeric(split_values)
            print(numeric_values)
            PIP_value = as.numeric(pip[c(numeric_values)])
            print(PIP_value)
            my.res = assoc_2[c(numeric_values),]
            my.res = cbind(my.res, PIP_value)
            my.res[,'group'] = i
            finemap_analysis = rbind(finemap_analysis,my.res)
    }}}
file_name = sprintf("../Results/CAU_MAF005_%s_finemap_result_window1000000_withPIP.txt",pheno)
print(file_name)
write.table(finemap_analysis,file_name, quote=F,row.names=F,col.names=T,sep="\t")
}

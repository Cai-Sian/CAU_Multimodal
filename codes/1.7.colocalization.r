#!/bin/Rscript

# colocolization
############################################

library(data.table)
library(coloc)

assoc <- sapply(c("PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"),function(x) fread(sprintf("../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_%s_window1000000.txt",x)),simplify=F)

############################
data_overlap <- read.table("../genotype_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_pooled_window1000000.txt",sep="\t",header=T)
print(unique(data_overlap$total_gp))
df_result_save = data.frame()
collect_all_result = data.frame()
for (section in unique(data_overlap$total_gp)){
    print(section)
    temp = data_overlap[data_overlap$total_gp == section,]
    print('snp in section:')
    print(nrow(temp))

    columns_sel <- grep('group_', names(temp), value = TRUE)
    df_specific <- temp[, columns_sel, drop = FALSE]
    all_empty <- sapply(df_specific, function(col) all(col == ""))
    notempty_columns <- names(df_specific)[!all_empty]
    # Print column names
    print(notempty_columns)


    if (length(notempty_columns)<2)
    {
        print('single signal found!')
    }
    else
    {   LDs <- read.table(sprintf("../genotype_analysis_inputs/Coloc_and_finemapping/LD_result/CAU_MAF005_window1000000_%s_r.ld",section), header = FALSE, sep = "\t", stringsAsFactors = FALSE)
        LDs <- as.matrix(LDs)
        snp_name = read.table(sprintf("../genotype_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_window1000000_%s_snp_list.txt",section), header = FALSE, sep = "\t", stringsAsFactors = FALSE)
        colnames(LDs) = snp_name$V1
        rownames(LDs) = snp_name$V1
     
        stripped_list <- gsub("^group_", "", notempty_columns)
        combinations_2 <- combn(stripped_list, 2, simplify = FALSE)

        length(combinations_2)
        for (i in combinations_2){
            print(i)

            pheno_1 = i[1]
            assoc_1 = assoc[[pheno_1]]
            #assoc_1 = assoc_1[p<1e-5]
            assoc_2 = assoc_1[(assoc_1$position>=min(temp$position))&(assoc_1$position<=max(temp$position)),]
            LD = LDs[assoc_2$snp,assoc_2$snp]
            if ((pheno_1 == 'plaque')|(pheno_1 == 'diagnosis'))
                {
                dataset1=list(snp=assoc_2$snp,position=assoc_2$position,beta=assoc_2$beta,pvalues=assoc_2$p,varbeta=(assoc_2$SE)*(assoc_2$SE),type=unique(assoc_2$type),N=unique(assoc_2$N),LD = LD,MAF=assoc_2$MAF)
                }

            else
                {
                dataset1=list(snp=assoc_2$snp,position=assoc_2$position,beta=assoc_2$beta,pvalues=assoc_2$p,varbeta=(assoc_2$SE)*(assoc_2$SE),type=unique(assoc_2$type),N=unique(assoc_2$N),sdY=unique(assoc_2$sdY),LD = LD)                
                }
            S3=runsusie(dataset1)
            
            pheno_2 = i[2]
            assoc_1_2 = assoc[[pheno_2]]
            #assoc_1_2 = assoc_1_2[p<1e-5]
            assoc_2_2 = assoc_1_2[(assoc_1_2$position>=min(temp$position))&(assoc_1_2$position<=max(temp$position)),]
            if ((pheno_2 == 'plaque')|(pheno_2 == 'diagnosis'))
                {
                dataset2=list(snp=assoc_2_2$snp,position=assoc_2_2$position,beta=assoc_2_2$beta,pvalues=assoc_2_2$p,varbeta=(assoc_2_2$SE)*(assoc_2_2$SE),type=unique(assoc_2_2$type),N=unique(assoc_2_2$N),MAF=assoc_2_2$MAF,LD = LD)
                }
            else
                {
                dataset2=list(snp=assoc_2_2$snp,position=assoc_2_2$position,beta=assoc_2_2$beta,pvalues=assoc_2_2$p,varbeta=(assoc_2_2$SE)*(assoc_2_2$SE),type=unique(assoc_2_2$type),N=unique(assoc_2_2$N),sdY=unique(assoc_2_2$sdY),LD = LD)
                }

            S4=runsusie(dataset2)
            my.res=coloc.susie(S3,S4)
            if (length(my.res$summary)==0){
                print('no colocalization')}
            else {
                if(requireNamespace("susieR",quietly=TRUE))
                {
                    for (idx in length(transpose(my.res$summary)))
                    {
                        print(idx)
                        sensitivity(my.res,"H4 > 0.8",row=idx,dataset1=dataset1,dataset2=dataset2)
                    }
                }
                
            o <- order(my.res$results$SNP.PP.H4.abf,decreasing=TRUE)
            cs <- cumsum(my.res$results$SNP.PP.H4.abf[o])
            w <- which(cs > 0.95)[1]
            co_snp = my.res$results[o,][1:w,]$snp
            
            co_snp <- paste(co_snp, collapse = ";")
            df_summary = data.frame(my.res$summary)
            df_summary['pheno1'] = pheno_1
            df_summary['pheno2'] = pheno_2
            df_summary['co_snp'] = co_snp
            df_summary['group'] = section
            df_result_save = rbind(df_result_save,df_summary)

            df_summary_all_tmp = data.frame(my.res$results)
            df_summary_all_tmp['pheno1'] = pheno_1
            df_summary_all_tmp['pheno2'] = pheno_2
            df_summary_all_tmp['group'] = section
            collect_all_result = rbind(collect_all_result,df_summary_all_tmp)
            }
        }
    }
}

write.table(df_result_save,"../Results/CAU_MAF005_coloc.susie_result_window1000000_summary.txt", quote=F,row.names=F,col.names=T,sep="\t")
write.table(collect_all_result,"../Results/CAU_MAF005_coloc.susie_result_window1000000_all.txt", quote=F,row.names=F,col.names=T,sep="\t")

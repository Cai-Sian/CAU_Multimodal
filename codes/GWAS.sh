##### GWAS (plink2) #####

texts=("PS" "MD" "ED" "TAMAX" "PI" "RI" "plaque" "diameter" "IMT" "diagnosis")

mkdir -p ../Results/GWAS_result

for text in "${texts[@]}"; do
    echo "$text"

    if [[ "$text" == "plaque" || "$text" == "diagnosis" ]]; then
        plink2 --bfile ../genetic_analysis_inputs/TWB_genotype_CAU \
            --pheno ../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize_cata.txt \
            --pheno-name $text \
            --covar ../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize_cata.txt \
            --covar-name SEX,AGE,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
            --covar-variance-standardize AGE PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10 \
            --logistic omit-ref hide-covar \
            --require-pheno $text \
            --maf 0.05 \
            --threads 64 \
            --out ../Results/GWAS_result/CAU_average_${text}_MAF005

    else
        plink2 --bfile ../genetic_analysis_inputs/TWB_genotype_CAU \
            --pheno ../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize.txt \
            --pheno-name $text \
            --covar ../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize.txt \
            --covar-name SEX,AGE,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
            --covar-variance-standardize AGE PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10 \
            --glm omit-ref hide-covar \
            --require-pheno $text \
            --threads 64 \
            --maf 0.05 \
            --out ../Results/GWAS_result/CAU_average_${text}_MAF005
    fi

done

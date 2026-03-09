# heritability (GCTA)
gcta64 --bfile ../genetic_analysis_inputs/TWB_genotype_CAU \
  --autosome \
  --maf 0.01 \
  --make-grm \
  --out ../genetic_analysis_inputs/heritability/TWB_genotype_CAU_grm \
  –-thread-num 100

# remove related individual
gcta64 --grm ../genetic_analysis_inputs/heritability/TWB_genotype_CAU_grm \
  --keep ../genetic_analysis_inputs/heritability/CAU_srm_sample.txt \
  --make-grm \
  --out ../genetic_analysis_inputs/heritability/TWB_genotype_CAU_grm_unrelated


####
GRM="../genetic_analysis_inputs/heritability/TWB_genotype_CAU_grm_unrelated"
INPUT_DIR="../genetic_analysis_inputs/heritability/sample_files"
OUTPUT_DIR="../Result/GCTA_results"

mkdir -p ${OUTPUT_DIR}


# Quantitive
PHENOS=("PS" "MD" "ED" "TAMAX" "PI" "RI" "diameter" "IMT")

for i in "${PHENOS[@]}"; do
    echo "========================================"
    echo "Running GCTA for phenotype: ${i}"
    echo "========================================"
    
    gcta64 --grm ${GRM} \
            --pheno ${INPUT_DIR}/pheno_${i}.txt \
            --qcovar ${INPUT_DIR}/qcov_${i}.txt \
            --covar ${INPUT_DIR}/cov_${i}.txt \
            --reml \
            --thread-num 96 \
            --out ${OUTPUT_DIR}/h2_${i}
    
    echo "Finished: ${i}"
    echo ""
done

echo "Quantitive analyses complete!"

# Binary
PHENOS=("diagnosis" "plaque")

# Set prevalence for each phenotype
# diagnosis-->calculate by the own data since cannot find from previous study (6416/(13467+6416)) = 0.323
# plaque --> take PMID:34674902
declare -A PREV
PREV["diagnosis"]=0.323
PREV["plaque"]=0.344


for i in "${PHENOS[@]}"; do
    echo "========================================"
    echo "Running GCTA for phenotype: ${i} (prevalence: ${PREV[$i]})"
    echo "========================================"
    
    gcta64 --grm ${GRM} \
            --pheno ${INPUT_DIR}/pheno_${i}.txt \
            --qcovar ${INPUT_DIR}/qcov_${i}.txt \
            --covar ${INPUT_DIR}/cov_${i}.txt \
            --reml \
            --prevalence ${PREV[$i]} \
            --thread-num 120 \
            --out ${OUTPUT_DIR}/h2_${i}
    
    echo "Finished: ${i}"
    echo ""
done

echo "All analyses complete!"

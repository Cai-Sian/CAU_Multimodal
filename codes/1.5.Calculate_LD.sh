#!/bin/bash
# create the ld
# extract SNP within window 1mb

# create LD_result folder if it doesn't exist
mkdir -p ../genetic_analysis_inputs/Coloc_and_finemapping/LD_result

# calculate LD score
file_list=(../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_*_window1000000_grp_*_snp_list.txt)

for i in "${file_list[@]}"; do
    echo $i
    output=$(basename "${i%_snp_list.txt}")
    plink --bfile ../genetic_analysis_inputs/TWB_genotype_all \
        --extract $i \
        --keep-allele-order \
        --r square \
        --out ../genetic_analysis_inputs/Coloc_and_finemapping/LD_result/${output}_r
done


#section 
file_list_2=(../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_*_window1000000_section_*_snp_list.txt)

for i in "${file_list_2[@]}"; do
    echo $i
    output_2=$(basename "${i%_snp_list.txt}")
    plink --bfile ../genetic_analysis_inputs/TWB_genotype_all \
        --extract "$i" \
        --keep-allele-order \
        --r square \
        --out ../genetic_analysis_inputs/Coloc_and_finemapping/LD_result/${output_2}_r
done

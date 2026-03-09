#!/bin/python3
# Genetic correlation

import pandas as pd
import gwaslab as gl

# Create summary statistics
def create_sumstate(pheno=None, input_file=None, binary=False, model='plink'):
    if binary == True:
        if model == 'plink':
            df = gl.Sumstats(input_file,
                     snpid="ID",
                     chrom="#CHROM",
                     pos="POS",
                     ea="A1",           
                     nea="REF",
                     OR="OR",
                     se="LOG(OR)_SE",
                     p="P",
                     n="OBS_CT", 
                     other=["Z_STAT"], 
                     build="38")
        df.basic_check(verbose=False)
        df.data.rename(columns={"Z_STAT": "Z"}, inplace=True)
    else:
        if model == 'plink':
            df = gl.Sumstats(input_file,
                 snpid="ID",
                 chrom="#CHROM",
                 pos="POS",
                 ea="A1", 
                 beta="BETA",
                 nea="REF",
                 se="SE",
                 p="P",
                 n="OBS_CT", 
                 build="38")
            df.basic_check(verbose=False)
    return pheno, df

## Create phenotype variables

pheno_names = ["PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"]
binary_phenos = ['diagnosis', 'plaque']

phenotypes = []
for phe in pheno_names:
    if phe in binary_phenos:
        phe_tmp = (phe, True, 'logistic.hybrid')
    else:
        phe_tmp = (phe, False, 'linear')
    phenotypes.append(phe_tmp)

base_path = "../Results/GWAS_result"

for pheno, binary, ext in phenotypes:
    input_file = f"{base_path}/CAU_average_${pheno}_MAF005.{pheno}.glm.{ext}"
    
    globals()[f'{pheno}_string'], globals()[pheno] = create_sumstate(
        pheno=pheno,
        input_file=input_file,
        binary=binary,
        model='plink'
    )
    print(f"Loaded: {pheno}")

## pairwise genetic correlation analysis
ref_ld_chr = "../genetic_analysis_inputs/genetic_correlation/ldscore_eas/eas_ldscores/"
w_ld_chr = "../genetic_analysis_inputs/genetic_correlation/ldscore_eas/eas_ldscores/"

results_rg = {}
df_results = pd.DataFrame()

for i, pheno in enumerate(pheno_names[:-1]):  # exclude last one
    other_pheno_names = pheno_names[i+1:]
    other_traits = [globals()[p] for p in other_pheno_names]
    rg_string = ",".join([globals()[f'{pheno}_string']] + [globals()[f'{p}_string'] for p in other_pheno_names])
    
    # Run the analysis
    result = globals()[pheno].estimate_rg_by_ldsc(
        other_traits=other_traits,
        rg=rg_string,
        ref_ld_chr=ref_ld_chr,
        w_ld_chr=w_ld_chr
    )
    
    results_rg[pheno] = result
    df_results = pd.concat([df_results,pd.DataFrame(result)],axis=0)
    
    print(f"Completed: {pheno} vs {other_pheno_names}")

df_results.to_csv(f'{base_path}/CAU_average_genetic_correlation_results.txt',sep='\t',index=False)
print(df_results)
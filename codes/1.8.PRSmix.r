#!/bin/Rscript

# PRSmix

library(PRSmix)

harmonize_snpeffect_toALT(
	ref_file = "../genetic_analysis_inputs/GWAS/twb_MAF.afreq", 
	pgs_folder = "../genetic_analysis_inputs/PRS/PRSmix_weight_input/",
	pgs_list = "../genetic_analysis_inputs/PRS/PGS_candidate_scoreID.txt",
	snp_col = 1,
	a1_col = 2,
	beta_col = 3,
	isheader = TRUE,
	chunk_size = 20,
	ncores = 96,
	out = "../genetic_analysis_inputs/PRS/PRSmix_PGS_candidate_scoreID_weights.txt"
)

compute_PRS(
	geno = "../genetic_analysis_inputs/TWB_genotype_CAU_trainval_for_PRSmix",
	weight_file = "../genetic_analysis_inputs/PRS/PRSmix_PGS_candidate_scoreID_weights.txt",
	plink2_path = "plink2",
	start_col = 4,
	out = "../genetic_analysis_inputs/PRS/PRSmix_PGS_candidate_scoreID_weights_CAU_trainval"
)


combine_PRS(
	pheno_file = "../genetic_analysis_inputs/PRS/CAU_diag_pheno.txt",
	covariate_file = "../genetic_analysis_inputs/PRS/CAU_diag_cov.txt",
	score_files_list = c("../genetic_analysis_inputs/PRS/PRSmix_PGS_candidate_scoreID_weights_CAU_trainval.sscore"),
	trait_specific_score_file = "../genetic_analysis_inputs/PRS/diagnosis_PRS.txt", ## pseudo list for executing
	pheno_name = "diagnosis",
	isbinary = TRUE,
	out = "../Results/PRSmix/Diagnosis_result_",
	liabilityR2 = TRUE,
	IID_pheno = "IID",
	covar_list = c("AGE", "SEX", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"),
	cat_covar_list = c("SEX"),
	ncores = 10,
	is_extract_adjSNPeff = TRUE,
	original_beta_files_list = "../genetic_analysis_inputs/PRS/PRSmix_PGS_candidate_scoreID_weights.txt",
	train_size_list = NULL,
	power_thres_list = c(0.8,0.95),
	pval_thres_list = c(0.8,0.05,0.00015),
	read_pred_training = FALSE,
)
import numpy as np
import pandas as pd

MAF = pd.read_csv('../genetic_analysis_inputs/GWAS/twb_MAF.frq',sep='\s+')
MAF = MAF[['SNP','A1','A2','MAF']].rename(columns={'SNP':'snp','A1':'minor','A2':'majour'})

for i in ["PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"]:
    print(i)
    if i in ["plaque", "diagnosis"]:
        df_all_mean_1 = pd.read_csv('../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize_cata.txt',sep='\t')
        df_all_mean_1 = df_all_mean_1.dropna(subset=i)

        df = pd.read_csv(f'../Results/GWAS_result/CAU_average_{i}_MAF005.{i}.glm.logistic.hybrid',sep='\t')
        df['#CHROM'] = df['#CHROM'].astype(int)
        df['POS'] = df['POS'].astype(int)
        df = df.sort_values(['#CHROM','POS']).reset_index(drop=True)
        df['BETA'] = np.log(df['OR'])
        df['SE'] = (df['LOG(OR)_SE'])    
        df =df[df['TEST'] == 'ADD']
        df = df.dropna(how="any", axis=0)
        df = df.rename(columns={'#CHROM':'chr','POS':'pos','BETA':'beta','ID':'snp','P':'p'})
        N_num = len(df_all_mean_1)
        df['N_cass'] = (list(df_all_mean_1[i]).count(2))
        df['Var_Y'] = N_num*((list(df_all_mean_1[i]).count(1))/N_num)*((list(df_all_mean_1[i]).count(2))/N_num)
        print(N_num*((list(df_all_mean_1[i]).count(1))/N_num)*((list(df_all_mean_1[i]).count(2))/N_num))
        df['fdr_hit'],df['fdr_pvalue'] = multi.fdrcorrection(df['p'])
        df['type'] = 'cc'
        df['position'] = range(len(df))
        df['N'] = N_num
        print(N_num)
        df = df.merge(MAF,how='left')
        df['MAF'] = np.where(df['A1'] == df['minor'],df['MAF'],1-df['MAF'])
        df = df[["beta","SE","snp","position","type","N","MAF","chr","pos",'p','fdr_pvalue','Var_Y','N_cass']]
        chr_col = 7
        pos_col = 8
    else:
        df_all_mean_1 = pd.read_csv('../genetic_analysis_inputs/GWAS/CAU_average_pheCov_INT_normalize.txt',sep='\t')
        df_all_mean_1 = df_all_mean_1.dropna(subset=i)
        N_num = len(df_all_mean_1)
        print(N_num)

        df = pd.read_csv(f'../Results/GWAS_result/CAU_average_{i}_MAF005.{i}.glm.linear',sep='\t')
        
        df['SE'] = (df['SE'])  
        df['#CHROM'] = df['#CHROM'].astype(int)
        df['POS'] = df['POS'].astype(int)
        df = df.sort_values(['#CHROM','POS']).reset_index(drop=True)
        df =df[df['TEST'] == 'ADD']
        df = df.dropna(how="any", axis=0)
        df = df.rename(columns={'#CHROM':'chr','POS':'pos','BETA':'beta','ID':'snp','P':'p'})
        df['fdr_hit'],df['fdr_pvalue'] = multi.fdrcorrection(df['p'])
        df['type'] = 'quant'
        df['sdY'] = np.std(df_all_mean_1[i])
        df['Var_Y'] = (np.std(df_all_mean_1[i]))**2
        df['position'] = range(len(df))
        df['N'] = N_num
        df = df[["beta","SE","snp","position","type","Var_Y","chr","pos",'p','fdr_pvalue','N','sdY']]
        chr_col = 6
        pos_col = 7

    df_1 = df.copy()
    j = 1
    top_can = (df_1[df_1['p'] <5e-8].sort_values('p'))
    print(top_can)
    range_num = 1000000
    while len(top_can) > 0:
        top_chro = top_can.iloc[0,chr_col]
        top = top_can.iloc[0,pos_col]
        print(top_can.iloc[0])
        df_temp = df[(df['chr'] == top_chro)&(df['pos']>=top-range_num)&(df['pos']<top+range_num)]
        df_temp = df_temp.sort_values('pos')
        print(df_temp.iloc[0])
        print(df_temp.iloc[-1])
        df_lim = df[df['chr'] == top_chro].sort_values('pos')
        df_upper_lim = df_lim.iloc[-1,pos_col]
        df_lower_lim = df_lim.iloc[0,pos_col]
        if top-range_num<1:
            df.loc[(df['chr'] == top_chro)&(df['pos']>0)&(df['pos']<top+range_num), 'group']  = f'grp_{j}'
            df_1 =df_1[~((df_1['chr'] == top_chro)&(df_1['pos']>0)&(df_1['pos']<top+range_num))]
        else:
            df.loc[(df['chr'] == top_chro)&(df['pos']>=top-range_num)&(df['pos']<top+range_num), 'group']  = f'grp_{j}'
            df_1 =df_1[~((df_1['chr'] == top_chro)&(df_1['pos']>=top-range_num)&(df_1['pos']<top+range_num))]

        top_can= (df_1[df_1['p'] <5e-8].sort_values('p'))
        j += 1
    print(df.head())

    df.to_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_{i}_window{range_num}.txt',sep='\t',index=False)
##
# group
range_num = 1000000 ##1mb
df_all = pd.DataFrame()
for i in ["PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"]:
    print(i)
#     df = pd.read_csv(f'/data/home/cliao/TWB_array/20241127_self_merge/GWAS_result/coloc/20241017_CAU_MAF005_twb1_2_impu_combine_sameA1A2_{i}_susie_window{range_num}.txt',sep='\t')
    df = pd.read_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_{i}_window{range_num}.txt',sep='\t')
    df_dpNA = df.dropna(subset = 'group')
    
    for gp in df_dpNA['group'].unique():
        print(gp)
        df_sub = df_dpNA[df_dpNA['group'] ==gp]
#         df_sub[['snp']].to_csv(f'/data/home/cliao/TWB_array/20241127_self_merge/GWAS_result/coloc/LD/finemap/20241017_CAU_MAF005_twb1_2_impu_combine_sameA1A2_{i}_susie_window{range_num}_{gp}.txt',sep='\t',index=False,header=False)
        df_sub[['snp']].to_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_{i}_window{range_num}_{gp}_snp_list.txt',sep='\t',index=False,header=False)

# section
range_num = 1000000 ##1mb
df_all = pd.DataFrame()
for i in ["PS", "MD", "ED", "TAMAX", "PI", "RI", "plaque", "diameter", "IMT", "diagnosis"][1:2]:
    df = pd.read_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_{i}_window{range_num}.txt',sep='\t')
    # display(df.sort_values('position'))
    df_sub = df[['chr','pos','position','snp','group']].rename(columns={'group':f'group_{i}'})
    df_sub = df_sub.set_index(['chr','pos','position','snp'])
    df_all = pd.concat([df_all,df_sub],axis=1)
    
    print(i)

df_all = df_all.dropna(how = 'all')
df_all = df_all.reset_index()
df_all['diff'] = df_all['position'].diff()
df_all_1 = df_all[df_all['diff']!=1]
record=[]
for a, i in enumerate(list(df_all_1.index)):
    if a <1:
        continue

    else:
        start = df_all_1.iloc[a-1]['position']
        end = df_all.iloc[i-1]['position']
    record.append(tuple([start,end]))

# + final one    
start = df_all_1.iloc[a]['position']
end = df_all.iloc[-1]['position']
record.append(tuple([start,end]))

for i, (a,b) in enumerate(record):
    df_all.loc[((df_all['position']>=a)& (df_all['position']<=b)),'total_gp'] = f'section_{i}'
    
df_all = df_all.dropna(subset= 'total_gp')


df_all.to_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_pooled_window{range_num}.txt',sep='\t',index=False)
df_all[['snp']].to_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_pooled_window{range_num}_snp_list.txt',sep='\t',index=False,header=None)


for i in df_all['total_gp'].unique():
    df_all_LD_temp = df_all[df_all['total_gp'] == i]
    df_all_LD_temp[['snp']].to_csv(f'../genetic_analysis_inputs/Coloc_and_finemapping/CAU_MAF005_forLD_window{range_num}_{i}_snp_list.txt',sep='\t',index=False,header=False) # for LD calculation

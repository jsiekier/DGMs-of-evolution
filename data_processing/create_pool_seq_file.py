import pickle as pkl
from variables import fix_variables


def convert_to_pool_seq_file(animal_noise,work_from='mogon',chrom=''):
    #animal_noise='10_target_sameSF_link_gauss_nrand_90_50_noise22'
    out_file_2=fix_variables[work_from][animal_noise]['sync_process']#base_path+'linked_gauss_90_50_randD_01_ad_noise_100_100.pkl'#'linked_gauss_90_50D_01_ad_noise_200_200.pkl'
    data=pkl.load(open(out_file_2,'rb'))

    if chrom!='':
        allele_data,distance_data,coverage=data['alleles'][chrom],data['distances'][chrom],data['cov'][chrom]
    else:
        allele_data,distance_data,coverage=data['alleles'],data['distances'],data['cov']



    pool_seq_path=fix_variables[work_from][animal_noise]['sync']
    out_stream=open(pool_seq_path,'w')
    for d, line_data,cov in zip(distance_data,allele_data,coverage):
        line=line_data.flatten()
        cov_line=cov.flatten()
        str_line='2L\t'+str(d)+'\tA\t'
        for freq,c in zip(line,cov_line):
            cov_1=(freq*c).astype(int)
            cov_2 = ((1-freq) *c).astype(int)

            str_line+=(str(cov_1)+":"+str(cov_2)+":0:0:0:0\t")
        str_line=str_line[:-1]
        str_line+='\n'
        out_stream.write(str_line)
    out_stream.close()


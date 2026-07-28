import pickle as pkl
from variables import fix_variables
import numpy as np


base_to_idx = {'a': 0, 'A': 0, 't': 1, 'T': 1, 'c': 2, 'C': 2, 'g': 3, 'G': 3}
idx_to_base={0:'A',1:'T',2:'C',3:'G'}
def convert_to_pool_seq_file(animal_noise,work_from='mogon',chrom=''):
    out_file_2=fix_variables[work_from][animal_noise]['sync_process']
    data=pkl.load(open(out_file_2,'rb'))

    if chrom!='':
        allele_data,distance_data,coverage=data['alleles'][chrom],data['distances'][chrom],data['cov'][chrom]
        nucleotide_idx=data['nucleotide_idx'][chrom]
    else:
        allele_data,distance_data,coverage=data['alleles'],data['distances'],data['cov']
        nucleotide_idx = data['nucleotide_idx']



    pool_seq_path=fix_variables[work_from][animal_noise]['sync']
    out_stream=open(pool_seq_path,'w')

    for line_idx,(d, line_data,cov) in enumerate(zip(distance_data,allele_data,coverage)):
        line=line_data.flatten()
        cov_line=cov.flatten()
        str_line='2L\t'+str(d)+'\t'+idx_to_base[nucleotide_idx[line_idx][0]]+'\t'
        for freq,c in zip(line,cov_line):
            coverages=np.zeros(4)
            coverages[nucleotide_idx[line_idx][0]]=(freq*c).astype(int)
            coverages[nucleotide_idx[line_idx][1]] = ((1-freq) *c).astype(int)

            for entry in coverages:
                str_line += (str(entry)+':')
            str_line += "0:0\t"

        str_line=str_line[:-1]
        str_line+='\n'
        out_stream.write(str_line)
    out_stream.close()


import pickle as pkl
import random
import numpy as np
from variables import fix_variables


def random_nucleotide_flip( work_from='mogon' ,animals=['experimental'] ):
    for animal in animals:
        print(animal)
        random.seed(0)
        pkl_file=fix_variables[work_from][animal]["sync_process"]

        out_file=pkl_file#'/lustre/miifs01/project/m2_datamining/sync_data/'+animal+'flip.pkl'#'data/haplos/experimental_nucleotide_flip.pkl'
        data = pkl.load(open(pkl_file, 'rb'))

        alleles=data['alleles']
        nucleotide_idx_dict=data['nucleotide_idx']
        allele_data_new,nucleotide_idx_new={},{}

        all_flips=dict()
        for chrom in alleles.keys():
            allele_data=alleles[chrom]
            nucleotide_data=nucleotide_idx_dict[chrom]
            num_snps=allele_data.shape[0]
            switch_positions=list(sorted(random.sample(list(range(num_snps)),num_snps//2)))
            all_flips[chrom]=switch_positions

            allele_data[switch_positions]=1-allele_data[switch_positions]
            nukleotide_switch=np.stack([nucleotide_data[switch_positions,1],nucleotide_data[switch_positions,0]],axis=-1)
            nucleotide_data[switch_positions]=nukleotide_switch


            allele_data_new[chrom]=allele_data
            nucleotide_idx_new[chrom]=nucleotide_data


        data={'alleles':allele_data_new,
              'distances':data['distances'],
              'fine_tuning_indices':data['fine_tuning_indices'],
              'nucleotide_idx':nucleotide_idx_new,
              'flips':all_flips}


        pkl.dump(data,open(out_file,'wb'))
        print('done')


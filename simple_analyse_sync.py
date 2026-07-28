import numpy as np
import pickle
import argparse


from wf_model.estimate_params import estimate_sh
from variables import fix_variables


base_to_idx={'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}


def estimale_all_s(sync_file_name,num_train_reps,num_train_generations,num_generations, t, Ne,out_file,nukleotide_idx):
    out_stream=open(out_file,'w')

    data_stream=open(sync_file_name,'r')
    line_counter=0
    for line in data_stream:

        splitted=line.split('\t')
        chrom,pos,major_allele=splitted[:3]
        alignment=splitted[3:]
        major_allele_num=nukleotide_idx[line_counter][0]

        snp_freqs=np.zeros(shape=(num_train_reps,num_train_generations))
        snp_cov = np.zeros(shape=(num_train_reps, num_train_generations))

        for pool_num, pool in enumerate(alignment):
            rep_idx = pool_num // num_generations
            gen_idx = pool_num % num_generations

            if rep_idx < num_train_reps and gen_idx < num_train_generations:
                pool_split = pool.split(":")[:4]
                # sum_=0
                cov_sum=0
                for n_idx, e in enumerate(pool_split):
                    coverage = float(e)
                    cov_sum+=coverage
                snp_freqs[rep_idx, gen_idx]=float(pool_split[major_allele_num])/cov_sum
                snp_cov[rep_idx, gen_idx] = int(pool_split[major_allele_num])

        line_counter+=1
        estimated_s=estimate_sh(snp_freqs, t, Ne, haploid=False, h=0.5, N_ctraj=0, cov=snp_cov, approximate=True, method="LLS")
        out_stream.write(str(estimated_s['s'])+' '+chrom+' '+pos+' '+major_allele+'\n')
    out_stream.close()




def estimale_all_s_dict(sync_file_name,num_train_reps,num_train_generations,num_generations, t, Ne_dict,out_file,nukleotide_idx_dict,position_nums,alleles,transpose=False,num_train_it=0):
    out_stream=open(out_file,'w')

    data_stream=open(sync_file_name,'r')
    line_counter=0
    curr_chrom=""
    for line in data_stream:
        splitted=line.split('\t')
        chrom,pos,major_allele=splitted[:3]
        if not line_counter:
            curr_chrom=chrom
        elif curr_chrom!=chrom:
            curr_chrom=chrom
            line_counter=0
        Ne=Ne_dict[chrom][num_train_it]
        alignment=splitted[3:]
        major_allele_num=nukleotide_idx_dict[chrom][line_counter][0]

        snp_freqs=np.zeros(shape=(num_train_reps,num_train_generations))
        snp_cov = np.zeros(shape=(num_train_reps, num_train_generations))

        for pool_num, pool in enumerate(alignment):
            if transpose:
                rep_idx = pool_num % num_train_reps
                gen_idx = pool_num // num_train_reps
            else:
                rep_idx = pool_num // num_generations
                gen_idx = pool_num % num_generations

            if rep_idx < num_train_reps and gen_idx < num_train_generations:
                pool_split = pool.split(":")[:4]
                # sum_=0
                cov_sum=0
                for n_idx, e in enumerate(pool_split):
                    coverage = float(e)
                    cov_sum+=coverage
                snp_freqs[rep_idx, gen_idx]=float(pool_split[major_allele_num])/cov_sum
                snp_cov[rep_idx, gen_idx] = int(float(pool_split[major_allele_num]))

        line_counter+=1
        estimated_s=estimate_sh(snp_freqs, t, Ne, haploid=False, h=0.5, N_ctraj=0, cov=snp_cov, approximate=True, method="LLS")
        out_stream.write(str(estimated_s['s'])+' '+chrom+' '+pos+' '+major_allele+'\n')
    out_stream.close()



def parse_animal_args():
    """ Adds animal-related arguments to a shared parser """
    parser = argparse.ArgumentParser(description="Simulation Parameters")

    animal_args = {
        'animal': 'Animal name',
        'animal_no_noise': 'ground truth dataset',
    }

    for arg, help_text in animal_args.items():
        parser.add_argument(f'--{arg}', type=str, required=False, help=help_text)
    parser.add_argument('--num_train_it', default=0, type=int)
    parser.add_argument('--work_from', default='home', type=str)
    # Step 3: Parse known arguments first
    args, unknown = parser.parse_known_args()
    return args




if __name__ == '__main__':
    args=parse_animal_args()
    work_from=args.work_from
    no_noise=args.animal_no_noise
    meta_data_name=args.animal
    num_train_it=args.num_train_it

    meta_data=fix_variables[meta_data_name]
    out_file = fix_variables[work_from][meta_data_name]['estimated_s'][0][0]
    sync_file_name =fix_variables[work_from][meta_data_name]['sync']
    pkl_file_name=fix_variables[work_from][no_noise]['sync_process']

    data=pickle.load(open(pkl_file_name,'rb'))
    nukleotide_idx=data['nucleotide_idx']['2L']

    num_generations=meta_data['num_generations']
    num_train_reps=meta_data['num_train_reps'][0]
    num_train_generations=meta_data['num_train_generations'][0]
    Ne=meta_data['Ne'][0]
    step_size=meta_data['step_size']
    t=[i*step_size for i in range(num_train_generations)]

    if isinstance(meta_data['Ne'],dict):
        print(meta_data_name,meta_data['Ne'])
        print(out_file)

        estimale_all_s_dict(sync_file_name, num_train_reps, num_train_generations, num_generations, t, meta_data['Ne'], out_file,
                       data['nucleotide_idx'],data['distances'],data['alleles'],transpose=True,num_train_it=num_train_it)
    else:
        print('synthetic data')
        Ne = meta_data['Ne'][num_train_it]
        nukleotide_idx = data['nucleotide_idx']['2L']
        estimale_all_s(sync_file_name,num_train_reps,num_train_generations,num_generations, t, Ne,out_file,nukleotide_idx)





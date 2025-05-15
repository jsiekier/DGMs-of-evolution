import numpy as np
import pickle
import argparse


from wf_model.estimate_params import estimate_sh
from variables import fix_variables


base_to_idx={'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}


def estimale_all_s(sync_file_name,num_train_reps,num_train_generations,num_generations, t, Ne,out_file,nukleotide_idx):
    #r_s_estimations=read_estimated_s(r_s_estimation_file)
    out_stream=open(out_file,'w')


    # read the file line by line:
    data_stream=open(sync_file_name,'r')
    line_counter=0
    for line in data_stream:

        splitted=line.split('\t')
        chrom,pos,major_allele=splitted[:3]
        alignment=splitted[3:]
        #estimated_result=r_s_estimations[chrom][int(pos)]
        #major_allele_num=base_to_idx[major_allele]
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
                    # sum_+=coverage
                    #snp_line[rep_idx, gen_idx, n_idx] = coverage
                snp_freqs[rep_idx, gen_idx]=float(pool_split[major_allele_num])/cov_sum
                snp_cov[rep_idx, gen_idx] = int(pool_split[major_allele_num])

        # now
        #tmp=1-snp_freqs
        #mean_afc=np.mean(snp_freqs[:,-1]-snp_freqs[:,0])
        #
        #snp_freqs=1-snp_freqs
        line_counter+=1
        estimated_s=estimate_sh(snp_freqs, t, Ne, haploid=False, h=0.5, N_ctraj=0, cov=snp_cov, approximate=True, method="LLS")
        #print(mean_afc,estimated_s['s'],estimated_result)
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
    # Step 3: Parse known arguments first
    args, unknown = parser.parse_known_args()
    return args



if __name__ == '__main__':

    work_from='mogon'
    args=parse_animal_args()
    no_noise=args.animal_no_noise
    meta_data_name=args.animal



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


    estimale_all_s(sync_file_name,num_train_reps,num_train_generations,num_generations, t, Ne,out_file,nukleotide_idx)




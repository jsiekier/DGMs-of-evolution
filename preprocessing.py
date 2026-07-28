from data_processing.create_pool_seq_file import convert_to_pool_seq_file
from data_processing.random_nukleotide_flip import random_nucleotide_flip
from data_processing.sampling_noise import make_noise_file
from variables import fix_variables
import argparse
from data_processing.Data_loader import read_sync_file_bi_allelic



def parse_animal_args():
    """ Adds animal-related arguments to a shared parser """
    parser = argparse.ArgumentParser(description="Simulation Parameters")

    animal_args = {
        'animal': 'Animal name',
        'work_from': 'Work from paths',
    }

    for arg, help_text in animal_args.items():
        parser.add_argument(f'--{arg}', type=str, required=False, help=help_text)
    # Step 3: Parse known arguments first
    args, unknown = parser.parse_known_args()
    return args



def get_haplo_bases(haplo_file):
    file_stream = open(haplo_file, 'r')
    haplo_bases=[]
    for line in file_stream:
        splitted=line.split('\t')
        bases=splitted[3].split(' ')
        haplo_bases.append(bases)
    return haplo_bases


if __name__ == '__main__':
    args=parse_animal_args()
    animal=args.animal
    work_from = 'mogon'
    num_generations = fix_variables[animal]['num_generations']
    num_replicates = fix_variables[animal]['num_replicates']
    num_train_generations = fix_variables[animal]['num_generations']
    num_train_reps = fix_variables[animal]['num_replicates']


    sync_file = fix_variables[work_from][animal]['sync']
    out_file = fix_variables[work_from][animal]['sync_process']

    if 'targets' in fix_variables[work_from][animal]:
        chrom='2L'
        haplo_file=fix_variables[work_from][animal]['start_haplos']
        haplo_bases = get_haplo_bases(haplo_file)
        transpose=False
        read_sync_file_bi_allelic(sync_file, num_generations, num_replicates, num_train_generations, num_train_reps,
                                  out_file, haplo_bases, zip=False, transpose=transpose)
        animal_noise = animal + 'noise14'
        make_noise_file(animal, animal_noise, [100], [40], chrom=chrom)
        convert_to_pool_seq_file(animal_noise, chrom=chrom)
    else:
        haplo_bases = None
        transpose=True
        read_sync_file_bi_allelic(sync_file, num_generations, num_replicates, num_train_generations, num_train_reps,
                                  out_file, haplo_bases, zip=False, transpose=transpose)
        random_nucleotide_flip(work_from=work_from, animals=[animal])







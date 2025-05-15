from data_processing.create_pool_seq_file import convert_to_pool_seq_file
from data_processing.sampling_noise import make_noise_file
from variables import fix_variables
import argparse




work_from='mogon'
num_generations=16
num_replicates=10
num_train_generations=16
num_train_reps=10
chrom='2L'

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

    sync_file = fix_variables[work_from][animal]['sync']
    out_file = fix_variables[work_from][animal]['sync_process']
    haplo_file=fix_variables[work_from][animal]['start_haplos']

    animal_noise=animal+'noise14'#'noise25'#
    make_noise_file(animal, animal_noise, [100], [40], chrom=chrom)
    convert_to_pool_seq_file(animal_noise, chrom=chrom)

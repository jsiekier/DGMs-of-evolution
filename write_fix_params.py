
import argparse
from data_processing.python_doc_reader_writer import read_python_dict_save, write_python


def parse_file_args(parser):
    """ Adds file path arguments to a shared parser """
    file_args = {
        'sync': 'Path to sync file',
        'sync_process': 'Path to sync process file',
        'start_haplos': 'Path to start haplotypes',
        'targets': 'Path to targets file',
        'eval_positions': 'Path to eval positions file',
        'estimated_s': 'Path to estimated selection coefficients',
        'LD_example': 'Path to LD example file',
        'cmh': 'Path to CMH file'
    }

    for arg, help_text in file_args.items():
        parser.add_argument(f'--{arg}', type=str, required=False, help=help_text)

def parse_animal_args(parser):
    """ Adds animal-related arguments to a shared parser """
    animal_args = {
        'animal': 'Animal name',
        'work_from': 'Work from paths',
    }

    for arg, help_text in animal_args.items():
        parser.add_argument(f'--{arg}', type=str, required=False, help=help_text)

def parse_param_args(parser):
    """ Adds simulation parameter arguments to a shared parser """
    param_args = {
        'step_size': (int, 5, 'Step size'),
        'num_generations': (int, 16, 'Number of generations'),
        'num_replicates': (int, 10, 'Number of replicates'),
        'num_train_reps': (int, 10, 'Number of training replicates'),
        'max_gene_len': (int, 8000, 'Maximum gene length'),
        'num_train_generations': (int, 7, 'Number of training generations'),
        'Ne': (int, [], 'Effective population size list'),
        'Nes': (int, [], 'List of Ne*s'),
        'Ncensus': (int, 1000, 'Census population size'),
        'Nsampling': (int, 100, 'Number of sampled individuals'),
        'coverage': (int, 40, 'Coverage depth')
    }

    for arg, (dtype, default, help_text) in param_args.items():
        nargs = '*' if isinstance(default, list) else None
        parser.add_argument(f'--{arg}', type=dtype, default=default, help=help_text, nargs=nargs)

def parse_arguments():
    """ Parses arguments and separates them into 3 dictionaries """
    # Step 1: Create main parser
    parser = argparse.ArgumentParser(description="Simulation Parameters")

    # Step 2: Add all arguments
    parse_file_args(parser)
    parse_animal_args(parser)
    parse_param_args(parser)

    # Step 3: Parse known arguments first
    args, unknown = parser.parse_known_args()

    # Step 4: Extract separate dictionaries
    file_keys = {'sync', 'sync_process', 'start_haplos', 'targets', 'eval_positions', 'estimated_s', 'LD_example', 'cmh'}
    animal_keys = {'animal', 'work_from'}
    param_keys = set(vars(args).keys()) - file_keys - animal_keys  # Everything else

    file_args = {k: v for k, v in vars(args).items() if k in file_keys}
    animal_args = {k: v for k, v in vars(args).items() if k in animal_keys}
    param_args = {k: v for k, v in vars(args).items() if k in param_keys}

    return file_args, animal_args, param_args






if __name__ == '__main__':
    python_file_name="variables.py"
    fix_variables,var_name=read_python_dict_save(python_file_name)
    # Print results
    print(type(fix_variables))  # Should be <class 'dict'>
    # read new data entry:
    file_args, animal_args, param_args = parse_arguments()
    # extend dict:
    fix_variables[animal_args['work_from']][animal_args['animal']]=file_args
    fix_variables[animal_args['animal']]=param_args
    fix_variables[animal_args['work_from']][animal_args['animal']]['estimated_s'] = [[fix_variables[animal_args['work_from']][animal_args['animal']]['estimated_s']]]
    fix_variables[animal_args['animal']]['num_train_reps']=[fix_variables[animal_args['animal']]['num_train_reps']]
    fix_variables[animal_args['animal']]['num_train_generations']=[fix_variables[animal_args['animal']]['num_train_generations']]

    # write added dict to the python file:
    write_python(python_file_name,var_name,fix_variables)




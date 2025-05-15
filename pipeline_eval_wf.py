import argparse
from eval.evaluation import evaluate_wf_model


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--animal', default='uniform_01_pw0', type=str)
    arguments = argparser.parse_args()
    return arguments

if __name__ == '__main__':
    tmp_data_folder='data/'
    args=parse_args()
    animal=args.animal
    wf_save_path = tmp_data_folder + 'wf_data/wf_simulations_' + args.animal + '_' + str(0) + 'no_n_new_s_pos.pkl'
    evaluate_wf_model(wf_save_path, animal, work_from='mogon', sample_size=100)
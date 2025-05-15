from eval.find_evaluation_idx import get_eval_positions
from eval.calculate_metric import calculate_metric
from variables import fix_variables
from linkage.t_linkage import calculate_linkage_metric
import pickle as pkl
import argparse

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--animal_gt', default='uniform_01_pw0', type=str)
    arguments = argparser.parse_args()
    return arguments

if __name__ == '__main__':

    base_path = 'data/'
    work_from='mogon'
    args=parse_args()
    animal_gt=args.animal_gt
    print('process',animal_gt)
    selection_file = fix_variables[work_from][animal_gt]["targets"]
    pkl_file = fix_variables[work_from][animal_gt]["sync_process"]
    haplo_file=fix_variables[work_from][animal_gt]["start_haplos"]
    out_file=fix_variables[work_from][animal_gt]["eval_positions"]



    eval_idx_dict = get_eval_positions(selection_file, haplo_file, region_size=500, target_region=50,
                                       num_unselected_positions=9000, pkl_file=pkl_file)  # 500 50 100000

    print('finish 1')
    raw_linkage_data =calculate_linkage_metric(selection_file, eval_idx_dict, haplo_file, pkl_file)
    pkl.dump(eval_idx_dict,open(out_file,'wb'))
    result=calculate_metric(raw_linkage_data, use_distance=False)
    eval_idx_dict['result']=result
    pkl.dump(eval_idx_dict,open(out_file,'wb'))
    print('finish 2')

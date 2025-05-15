import pickle as pkl
import random
import numpy as np


random.seed(0)


def read_positions(file_path):
    file_stream=open(file_path,'r')
    positions=[]
    for line in file_stream:
        positions.append(int(line.split('\t')[1]))
    file_stream.close()
    return positions




def get_eval_positions(selection_file,haplo_file,region_size=500,target_region=50,num_unselected_positions=9000,pkl_file=''):



        if selection_file!='':
            haplo_positions = read_positions(haplo_file)
            selection_positions = read_positions(selection_file)

            unselected_idx = []
            selected_idx = []

            s_idx_last = -region_size
            for s_position in selection_positions:
                s_idx = haplo_positions.index(s_position)
                unselected_idx.extend(range(s_idx_last + region_size, s_idx - region_size))
                s_idx_last = s_idx
                selected_idx.extend(
                    range(max(0, s_idx - target_region), min(len(haplo_positions), s_idx + target_region)))
            unselected_sample = list(sorted(random.sample(unselected_idx, num_unselected_positions)))
        else:

            unselected_sample = dict()
            selected_idx = dict()
            data = pkl.load(open(pkl_file, 'rb'))
            #print('data load',flush=True)
            fine_tuning_indices =data['fine_tuning_indices']['2L']
            distance_data=data['distances']['2L']
            position_nums=num_unselected_positions//2
            num_all_positions=sum([entry[1].shape[0] for entry in distance_data.items()])
            #print('num all positions',num_all_positions,'num eval positions',position_nums,flush=True)

            for key in fine_tuning_indices.keys():
                chrom_positions=int(position_nums*(distance_data[key].shape[0]/num_all_positions))
                #print('chrom positions',chrom_positions,distance_data[key].shape[0],flush=True)
                fine_tuning=np.squeeze(fine_tuning_indices[key])
                selected_idx[key]=np.random.choice(fine_tuning,chrom_positions)
                none_fine_tuning=set(range(len(distance_data[key]))).difference(set(fine_tuning.tolist()))
                unselected_sample[key]=np.random.choice(list(none_fine_tuning),chrom_positions)
                #print(selected_idx[key],flush=True)
                #print(unselected_sample[key],flush=True)






        return {'selected':selected_idx,'unselected':unselected_sample}



from data_processing.choose_selection_points import declare_targets
from data_processing.create_artificial_haplos import create_haplo_array, write_file
import numpy as np

if __name__ == '__main__':
    haplo_file = 'data/mimicree_format_dgrp_snps_2L'
    file_end='r2_max_pw0_t10'
    target_file_out = 'data/targets/targets_'+file_end+'.txt'

    number_of_individuals=1000
    number_of_chom=2*number_of_individuals

    num_positions=910880
    haplo_arr_org=create_haplo_array(number_of_chom,num_positions=num_positions,min_freq=0.05,max_freq=0.95)#910880
    np.random.seed(0)
    for n_idx,(noise_level,noise_path) in enumerate(zip([0,0.04,0.08,0.12],['0','4','8','12'])):

        haplo_arr=np.copy(haplo_arr_org)
        haplo_file_out = 'data/haplos_max_'+noise_path+'_r'

        num_elements = haplo_arr.size
        num_to_flip = int(noise_level * num_elements)

        # Randomly select indices to flip
        if num_to_flip:
            indices = np.unravel_index(np.random.choice(num_elements, num_to_flip, replace=False), haplo_arr.shape)

            # Flip the values
            haplo_arr[indices] = 1 - haplo_arr[indices]
        # write the file:
        write_file(haplo_file, haplo_file_out,haplo_arr,num_positions=num_positions)

        for num_targets,effect_size in zip([10,25,50],[0.2,0.08,0.04]):
            target_file_out = 'data/targets/targets_max' + noise_path +'_t_'+str(num_targets)+ '.txt'
            declare_targets(haplo_file_out, target_file_out, num_targets, add_effect_distance=effect_size, min_freq=0.40,max_freq=0.60)


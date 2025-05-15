from data_processing.file_reader import read_positions
from t_linkage import read_haplo_file, calculate_all_linkages
from variables import fix_variables
import random
import os
import pickle as pkl
random.seed(0)



def calculate_LD_on_subset(animals,work_from,sample_size=1000,n_window=50,ld_idx=1):
    # 1. get position names
    #gt_animal1
    haplo_file=fix_variables[work_from][animals[0]]['start_haplos']
    positions=read_positions(haplo_file)


    # 2. sample 2 sets of random position combinations
    pos_idx=list(range(len(positions)-1))

    sample_direct_neighbor_idx=random.sample(pos_idx,sample_size)
    sample_direct_neighbor=[(positions[idx],positions[idx+1]) for idx in sample_direct_neighbor_idx]

    pos_idx=pos_idx[:-n_window]
    neighbor_add=list(range(2,n_window))
    sample_indirect_neighbor_idx = random.sample(pos_idx, sample_size)
    sample_indirect_neighbor = [(positions[idx], positions[idx + random.choice(neighbor_add)]) for idx in sample_indirect_neighbor_idx]

    # 3. calculate LD for all these combinations and save this in an LD directory
    all_LDs=[[] for _ in range(len(animals))]

    #for a_idx, (animal_arr,_) in enumerate(animals):
    for a_idx,animal_gt in enumerate(animals):
        #animal_gt=animal_arr[0]
        LD_file_path=fix_variables[work_from][animal_gt]['LD_example']
        if not os.path.isfile(LD_file_path):

            #3.1
            # read haplo file
            haplo_file = fix_variables[work_from][animal_gt]['start_haplos']
            haplo_data = read_haplo_file(haplo_file,nucleotide_idx=[])
            print('read haplo finished')


            for pair_set in [sample_direct_neighbor,sample_indirect_neighbor]:
                LDs = []
                for (n,neighbor) in pair_set:
                    result = calculate_all_linkages(n, [neighbor],haplotype_dict=haplo_data)
                    LDs.append((result[0][0],result[1][0]))

                all_LDs[a_idx].append(LDs)

            # save the LD file
            pkl.dump(all_LDs[a_idx],open(LD_file_path,'wb'))
        else:
            LDs=pkl.load(open(LD_file_path,'rb'))
            all_LDs[a_idx]=LDs

    all_LDs_def=[]
    for entry in all_LDs:
        all_LDs_def.append([])
        for e in entry:
            defs =zip(*e)
            all_LDs_def[-1].append(list(list(defs)[ld_idx]))
    return  all_LDs_def




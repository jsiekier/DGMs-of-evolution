import random
import numpy as np

np.random.seed(0)





def create_haplo_array(number_of_chom,num_positions=2000,min_freq=0.15,max_freq=0.85):
    random.seed(0)
    np.random.seed(0)
    freqs = np.random.uniform(min_freq, max_freq, size=num_positions)
    arr=[]

    print(freqs)
    idx_list=[0,1]
    for i in range(num_positions):
        pB1_count=int(number_of_chom*freqs[i])
        pB2_count=number_of_chom-pB1_count
        random.shuffle(idx_list)
        arr.append([idx_list[0]]*pB1_count+[idx_list[1]]*pB2_count)
    return np.asarray(arr)


def write_file(haplo_file, haplo_file_out,haplo_arr,num_positions=-1 ):

    splitted_arr=[]
    haplo_in_stream = open(haplo_file,'r')
    haplo_stream_out=open(haplo_file_out,"w")
    for line in haplo_in_stream:
        splitted=line.split('\t')
        splitted_arr.append(splitted)
    print('num positions',len(splitted_arr))
    haplo_in_stream.close()

    if num_positions<0:
        end_pos=len(splitted_arr)
    else:
        end_pos=num_positions

    #line[1]
    for l_idx,line in enumerate(splitted_arr[:end_pos]):
        new_line = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3]+'\t'
        allele_pair = line[3].split('/')
        for h_idx, allele_idx in enumerate(haplo_arr[l_idx]):
            if h_idx % 2 == 0 and h_idx:
                new_line += ' '
            new_line += allele_pair[allele_idx]
        new_line += '\n'
        haplo_stream_out.write(new_line)
    haplo_stream_out.close()









import pickle
import numpy as np
import os
import math
def read_positions(file_path):
    file_stream=open(file_path,'r')
    positions=[]
    for line in file_stream:
        positions.append(int(line.split('\t')[1]))
    file_stream.close()
    return positions
def read_estimated_s(estimated_s_path):
    base_to_idx = {'a': 0, 'A': 0, 't': 1, 'T': 1, 'c': 2, 'C': 2, 'g': 3, 'G': 3}

    # check if path is a file or directory:
    estimated_s_files=[]
    if os.path.isdir(estimated_s_path):
        for f in os.listdir(estimated_s_path):
            estimated_s_files.append(estimated_s_path+f)
    else:
        estimated_s_files=[estimated_s_path]
    estimated_s=dict()
    for estimated_s_file in estimated_s_files:
        haplo_stream = open(estimated_s_file, 'r')

        i=0
        chrom=None
        for line in haplo_stream:
            i+=1
            k = line.replace('\n','').split(" ")
            if k[0]=="NA":
                s=0
            else:
                # take the negative value as pool-seq computes s for the minor allele and we look at the major allele
                s=-float(k[0])
                if s <-1 or s>1:
                    s=0

            chrom=k[1]
            #minor_nucleotide=k[3]

            position=int(k[2])#-1
            if chrom not in estimated_s:
                estimated_s[chrom]=dict()
            estimated_s[chrom][position]=s#,base_to_idx[minor_nucleotide]]
        print('num lines:',i)
    return estimated_s


def read_targets(selection_pos_file,sync_file):
    data_stream=open(selection_pos_file,'r')
    normal_pos=[]
    for line in data_stream:
        splitted=line.split('\t')
        normal_pos.append(int(splitted[1]))
    data = pickle.load(open(sync_file, 'rb'))
    if isinstance(data['alleles'], dict):
        positions = data['distances']['2L']
    else:
        positions = data['distances']


    targets=np.flatnonzero(np.isin(positions, normal_pos))
    no_targets=np.flatnonzero(~np.isin(positions, normal_pos))

    targets_org_pos = positions[targets]
    no_targets_org_pos = positions[no_targets]
    return targets, no_targets,targets_org_pos,no_targets_org_pos

def read_estimated_s_positions(estimated_s_file,positions_arr):
    s_dict=read_estimated_s(estimated_s_file)
    s_dict_chom=s_dict['2L']
    result=[[],[]]
    for i,pos in enumerate(positions_arr):
        for p in pos:
            result[i].append(s_dict_chom[p])
    return result










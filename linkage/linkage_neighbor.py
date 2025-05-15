import seaborn as sns
import pandas as pd
import numpy as np
import gzip
from matplotlib import pyplot as plt



def read_haplo_file(input_file_name,is_gzip=1):
    haplotype_dict = dict()
    if is_gzip:
        with gzip.open(input_file_name, 'rb') as f:  # open(input_file_name, 'r') as f:#
            for line in f:
                line = line.decode('utf-8')
                if not line.startswith('#'):
                    k = line.split("\t")
                    chrm = k[0]
                    pos = int(k[1])  # -1
                    haplotypes1 = k[4].split()
                    haplotypes1 = ''.join(haplotypes1)
                    characters = list(set(haplotypes1))
                    if len(characters) == 2:
                        haplotypes1 = haplotypes1.replace(characters[0], '0').replace(characters[1], '1')
                    else:
                        haplotypes1 = haplotypes1.replace(characters[0], '0')
                    haplotype_dict[pos]=((haplotypes1,haplotypes1.count('0')/len(haplotypes1)))
    else:
        with open(input_file_name, 'r') as f:  # open(input_file_name, 'r') as f:#
            for line in f:
                if not line.startswith('#'):
                    k = line.split("\t")
                    chrm = k[0]
                    pos = int(k[1])  # -1
                    haplotypes1 = k[4].split()
                    haplotypes1 = ''.join(haplotypes1)
                    characters = list(set(haplotypes1))
                    if len(characters) == 2:
                        haplotypes1 = haplotypes1.replace(characters[0], '0').replace(characters[1], '1')
                    else:
                        haplotypes1 = haplotypes1.replace(characters[0], '0')
                    haplotype_dict[pos]=((haplotypes1,haplotypes1.count('0')/len(haplotypes1)))

    return haplotype_dict


def calculate_all_linkages( position, neighbors, haplotype_dict):
    linkages = []

    haplotypes2 = haplotype_dict[position][0]
    haplotypes2_neg = haplotypes2.replace('0', 'X').replace('1', '0').replace('X', '1')
    neighbors_filtered = []
    for idx_n in neighbors:
        if position != idx_n:
            haplotypes1 = haplotype_dict[idx_n][0]

            if len(haplotypes1) != len(haplotypes2):
                print('Error', len(haplotypes1), len(haplotypes2))
                print(haplotypes1)
                print(haplotypes2)
            else:
                neighbors_filtered.append(idx_n)
                linkage_possibilities = []
                for other_haplo in [haplotypes2, haplotypes2_neg]:
                    A1B1 = 0
                    A2B2 = 0
                    A1B2 = 0
                    A2B1 = 0

                    for i in range(0, len(haplotypes1)):

                        if haplotypes1[i] == "0" and other_haplo[i] == "0":
                            A1B1 += 1
                        elif haplotypes1[i] == "0" and other_haplo[i] == "1":
                            A2B1 += 1
                        elif haplotypes1[i] == "1" and other_haplo[i] == "0":
                            A1B2 += 1
                        elif haplotypes1[i] == "1" and other_haplo[i] == "1":
                            A2B2 += 1

                    freqsA1B1 = A1B1 / float(len(haplotypes1))
                    freqsA2B2 = A2B2 / float(len(haplotypes1))

                    freqsA1B2 = A1B2 / float(len(haplotypes1))
                    freqsA2B1 = A2B1 / float(len(haplotypes1))

                    D = (freqsA1B1 * freqsA2B2) - (freqsA1B2 * freqsA2B1)
                    p1 = haplotypes1.count('0') / len(haplotypes1)
                    p2 = 1 - p1
                    q1 = other_haplo.count('0') / len(other_haplo)
                    q2 = 1 - q1
                    divisor = (p1 * p2 * q1 * q2)
                    if divisor:
                        rr = (D * D) / divisor
                    else:
                        rr = D
                    if D > 0:
                        D /= min([p1 * q2, p2 * q1])
                    elif D < 0:
                        D /= max([-p1 * q1, -p2 * q2])

                    linkage_possibilities.append([D,rr])#rr
                linkages.append(linkage_possibilities)
    if not len(linkages):
        return np.nan,np.nan,np.nan,np.nan
    linkages = np.asarray(linkages)
    linkages = np.max(np.abs(linkages), axis=1)
    linkage_mean = np.mean(linkages, axis=0)
    linkage_max=np.max(linkages, axis=0)
    return [linkage_mean[0],linkage_max[0],linkage_mean[1],linkage_max[1]]


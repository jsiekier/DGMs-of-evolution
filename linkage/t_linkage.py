import pickle as pkl
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import time
import matplotlib


matplotlib.rcParams['lines.markersize'] = 5


def calculate_all_linkages_nans(position, neighbors, haplotype_dict):

    linkages = []
    linkage_trust = []

    haplotypes2, len_haplo2, num_char_0_2, num_char_1_2, num_nan_2 = haplotype_dict[position]


    for idx_n in neighbors:

        if position != idx_n:

            haplotypes1, len_haplo1, num_char_0_1, num_char_1_1, num_nan_1 = haplotype_dict[idx_n]


            haplo_zips = list(zip(haplotypes1, haplotypes2))
            # count similar tuples
            counter = Counter(haplo_zips)
            num_real_haplos=counter[(0, 0)]+counter[(1, 1)]+counter[(0, 1)]+counter[(1, 0)]
            #linkage_trust.append(min(1 - (num_nan_2 / len_haplo2), 1 - (num_nan_1 / len_haplo1)))
            linkage_trust.append(num_real_haplos/len_haplo1)
            D = 0
            if num_real_haplos:

                freqsA1B1 = counter[(0, 0)] / num_real_haplos
                freqsA2B2 = counter[(1, 1)] / num_real_haplos

                freqsA1B2 = counter[(0, 1)] / num_real_haplos
                freqsA2B1 = counter[(1, 0)] / num_real_haplos

                D = (freqsA1B1 * freqsA2B2) - (freqsA1B2 * freqsA2B1)
                p1 = (counter[(0, 1)]+counter[(0, 0)])/num_real_haplos
                p2 = (counter[(1, 1)]+counter[(1, 0)])/num_real_haplos
                q1 = (counter[(0, 0)]+counter[(1, 0)])/num_real_haplos
                q2 = (counter[(1, 1)]+counter[(0, 1)])/num_real_haplos

                divisor=0
                if D > 0:
                    divisor=min([p1 * q2, p2 * q1])

                elif D < 0:
                    divisor=max([-p1 * q1, -p2 * q2])
                if divisor:
                    D /= divisor
                else:
                    D=0

            linkages.append(D)
        else:
            linkages.append(1.0)
    if not len(linkages):
        print('Error')
        return []

    return linkages, linkage_trust

def calculate_all_linkages( position, neighbors, haplotype_dict):
    linkages = []
    r_square=[]
    normal_D=[]
    filter_positions=[]
    haplotypes2,freq2 = haplotype_dict[position]




    for idx_n in neighbors:
        filter_positions.append(idx_n)
        if position != idx_n:

            haplotypes1,freq1 = haplotype_dict[idx_n]
            haplo_zips=list(zip(haplotypes1,haplotypes2))
            #print('fn',freq2,freq1, haplo_zips)

            # count similar tuples
            counter = Counter(haplo_zips)

            freqsA1B1 = counter[(0,0)] / len(haplotypes1)
            #freqsA2B2 = counter[(1,1)]/ len(haplotypes1)
            #freqsA1B2 = counter[(0,1)] / len(haplotypes1)
            #freqsA2B1 = counter[(1,0)] / len(haplotypes1)
            p1 = 1-freq1
            p2 = freq1
            q1 = 1-freq2
            q2 = freq2
            D=freqsA1B1-(p1*q1)
            #D = (freqsA1B1 * freqsA2B2) - (freqsA1B2 * freqsA2B1)
            #D_tmp=D

            divisor = (p1 * p2 * q1 * q2)
            normal_D.append(D)
            if divisor:
                rr = (D * D) / divisor
            else:
                rr = D
            if D > 0:
                tmp1=[p1 * q2, p2 * q1]
                D /= min(tmp1)
            elif D < 0:
                tmp1=[-p1 * q1, -p2 * q2]
                D /= max(tmp1)


            r_square.append(rr)
            linkages.append(D)
        else:
            r_square.append(1.0)
            normal_D.append(1.0)
            linkages.append(1.0)

    if not len(linkages):
        print('Error')
        return []

    return linkages,r_square,normal_D,filter_positions
base_to_idx={'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}
nukleotide_to_char={0:'A',1:'T',2:'C',3:'G'}

def read_haplo_file(input_file_name,nucleotide_idx=[]):
    print(nucleotide_idx,len(nucleotide_idx))
    if  len(nucleotide_idx)==0:
        return read_haplo_file_no_nucleotides(input_file_name)
    haplotype_dict = dict()
    start_time=time.time()
    line_counter=0
    with open(input_file_name, 'r') as f:  # open(input_file_name, 'r') as f:#
        for line in f:
            if not line.startswith('#'):
                k = line.split("\t")
                chrm = k[0]
                pos = int(k[1])  # -1
                haplotypes1 = k[4].split()
                haplotypes1 = ''.join(haplotypes1)
                nukleotide_1,nukleotide_2=nucleotide_idx[line_counter]
                characters=[nukleotide_to_char[nukleotide_1],nukleotide_to_char[nukleotide_2]]
                #characters = list(set(haplotypes1))
                if len(characters) == 2:
                    haplotypes1 = haplotypes1.replace(characters[0], '0').replace(characters[1], '1')
                else:
                    haplotypes1 = haplotypes1.replace(characters[0], '0')

                #haplotypes1=[int(char) for char in haplotypes1]
                haplotypes1 =list(map(int,haplotypes1))
                haplotype_dict[pos]=((haplotypes1,sum(haplotypes1)/len(haplotypes1)))
                line_counter+=1
    print((time.time() - start_time) / 60, 'time for haplo reading')

    return haplotype_dict
def read_haplo_file_no_nucleotides(input_file_name):
    haplotype_dict = dict()
    start_time=time.time()
    line_counter=0
    with open(input_file_name, 'r') as f:  # open(input_file_name, 'r') as f:#
        for line in f:
            if not line.startswith('#'):
                k = line.split("\t")
                chrm = k[0]
                pos = int(k[1])  # -1
                haplotypes1 = k[4].split()
                haplotypes1 = ''.join(haplotypes1)
                #nukleotide_1,nukleotide_2=nucleotide_idx[line_counter]
               # characters=[nukleotide_to_char[nukleotide_1],nukleotide_to_char[nukleotide_2]]
                characters = list(set(haplotypes1))
                if len(characters) == 2:
                    haplotypes1 = haplotypes1.replace(characters[0], '0').replace(characters[1], '1')
                else:
                    haplotypes1 = haplotypes1.replace(characters[0], '0')

                #haplotypes1=[int(char) for char in haplotypes1]
                haplotypes1 =list(map(int,haplotypes1))
                haplotype_dict[pos]=((haplotypes1,sum(haplotypes1)/len(haplotypes1)))
                line_counter+=1
    print((time.time() - start_time) / 60, 'time for haplo reading')

    return haplotype_dict

def read_haplo_file_chrom(input_file_name,nukleotide_idx,positions):
    idx_to_base = {0:'A',1: 'T', 2: 'C',3: 'G'}
    haplotype_dict = dict()
    start_time=time.time()
    nan_debug_plot_info=[]
    all_chroms=[]
    with open(input_file_name, 'r') as f:  # open(input_file_name, 'r') as f:#
        for line in f:
            if not line.startswith('#'):
                k = line.split("\t")
                chrom = k[0]
                if chrom not in haplotype_dict:
                    haplotype_dict[chrom]=dict()
                    nan_debug_plot_info.append([])
                    all_chroms.append(chrom)
                pos = int(k[1])  # -1
                pos_in_numpy=np.where(positions[chrom]==pos)[0][0]
                characters=[idx_to_base[nukleotide_idx[chrom][pos_in_numpy][0]],
                            idx_to_base[nukleotide_idx[chrom][pos_in_numpy][1]]]
                characters_set=set(characters)
                characters_set.add('N')

                haplotypes1 = k[4].split()
                haplotypes1 = ''.join(haplotypes1)
                replacement=0
                characters_remaining = set(haplotypes1).difference(characters_set)
                for char in characters:
                    haplotypes1 = haplotypes1.replace(char, str(replacement))
                    replacement+=1

                haplotypes1 = haplotypes1.replace('N', str(2))
                for c_idx,char in enumerate(characters_remaining):
                    haplotypes1 = haplotypes1.replace(char, str(3+c_idx))

                #else:
                #    haplotypes1 = haplotypes1.replace(characters[0], '0')

                #haplotypes1=[int(char) for char in haplotypes1]
                len_haplo=len(haplotypes1)
                num_nan=haplotypes1.count('2')
                num_char_0=haplotypes1.count('0')
                num_char_1=haplotypes1.count('1')


                haplotypes1 =list(map(int,haplotypes1))
                haplotype_dict[chrom][pos]=((haplotypes1,len_haplo,num_char_0,num_char_1,num_nan))
                nan_debug_plot_info[-1].append(num_nan/len_haplo)
    print((time.time() - start_time) / 60, 'time for haplo reading')
    ###############################################################
    # short debugging_plot:
    fig, axis= plt.subplots(3, 2, figsize=(5, 7))
    for chrom_idx, chrom in enumerate(all_chroms):
        i=chrom_idx//2
        j=chrom_idx%2
        sns.histplot(nan_debug_plot_info[chrom_idx], ax=axis[i, j])
        axis[i, j].set_title('Chrom: ' + chrom)
    fig.tight_layout()
    fig.savefig('plots/nan_analysis.png')

    return haplotype_dict



def read_selection_positions(selection_file_name):
    selected_positions = dict()
    selection_file_stream = open(selection_file_name, 'r')
    #chrom = None
    for line in selection_file_stream:
        splitted = line.split('\t')
        chrom, position = splitted[:2]
        effect=splitted[3]
        selected_positions[int(position)] = float(effect)

    selection_file_stream.close()
    return selected_positions


def read_afc(pkl_file,chrom='2L'):
    data = pkl.load(open(pkl_file, 'rb'))
    if isinstance(data['alleles'],dict):
        allele_data, positions= data['alleles'][chrom], data['distances'][chrom]
        nucleotide_idx=data['nucleotide_idx'][chrom]
    else:
        allele_data, positions= data['alleles'], data['distances']
        nucleotide_idx=data['nucleotide_idx']
    # shape(num snps,generations,replicates)
    #mean_AFC=np.mean(np.abs(allele_data[:,:,0]-allele_data[:,:,4]),axis=-1)
    mean_AFC = np.mean(allele_data[:, :, 0] - allele_data[:, :, 5],axis=1)
    return mean_AFC,positions,nucleotide_idx


def read_afc_chrom(pkl_file,rep,afc_generation=3):


    data = pkl.load(open(pkl_file, 'rb'))
    allele_data, positions,nukleotide_idx= data['alleles'], data['distances'],data['nucleotide_idx']
    # shape(num snps,generations,replicates)
    pos_less=dict()
    mean_AFC=dict()
    sf=dict()
    for key in allele_data.keys():
        if rep>=0:
            #TODO
            mean_AFC[key] = np.abs(np.squeeze(allele_data[key])[:, rep, 0] -
                                          np.squeeze(allele_data[key])[:, rep, afc_generation])
            pos_less[key] = np.squeeze(positions[key])
            nukleotide_idx[key] = np.squeeze(nukleotide_idx[key])
            sf[key] = np.squeeze(allele_data[key])[:, rep, 0]
        else:
            mean_AFC[key]=np.max(np.abs(np.squeeze(allele_data[key])[:,:,0]-
                                         np.squeeze(allele_data[key])[:,:,afc_generation]),axis=-1)
            pos_less[key]=np.squeeze(positions[key])
            nukleotide_idx[key]=np.squeeze(nukleotide_idx[key])
            sf[key]=np.mean(np.squeeze(allele_data[key])[:,:,0],axis=-1)

    return mean_AFC,pos_less,nukleotide_idx,sf





def linkage_base(eval_positions_,haplo_file,pkl_file,window,rep):
    mean_AFC, positions, nukleotide_idx, sf = read_afc_chrom(pkl_file,rep)
    # read eval file to get eval positions:
    haplo_data_chrom = read_haplo_file_chrom(haplo_file, nukleotide_idx, positions)

    all_n_f = dict()
    all_n_f_distance = dict()
    all_n_afc = dict()
    linkage_trust_dict = dict()
    afc_filter = dict()
    sf_filter = dict()

    for key in haplo_data_chrom.keys():
        eval_positions = np.concatenate([eval_positions_['selected'][key], eval_positions_['unselected'][key]])
        # read haplotypes:
        haplo_data = haplo_data_chrom[key]
        all_n_f[key] = []
        all_n_f_distance[key] = []
        all_n_afc[key] = []
        linkage_trust_dict[key] = []

        gen_pos_to_arr_pos = dict()
        # print(key)
        for arr_pos, gen_pos in enumerate(positions[key]):
            gen_pos_to_arr_pos[gen_pos] = arr_pos

        for p_idx, pos in enumerate(eval_positions):
            if p_idx % 10 == 0:
                print('Linkage:', p_idx, 'out of', len(eval_positions), key, len(mean_AFC[key]))

            chrom_pos_focus = positions[key][pos]
            neighbor_pos = np.concatenate([positions[key][pos - window:pos], positions[key][pos + 1:pos + window + 1]],
                                          axis=0)

            neighbor_afc = []
            for n in neighbor_pos:
                neighbor_afc.append(mean_AFC[key][gen_pos_to_arr_pos[n]])
            all_n_afc[key].append(neighbor_afc)
            #print(all_n_afc[key][-1])
            link_x_s, linkage_trust = calculate_all_linkages_nans(chrom_pos_focus, neighbor_pos,
                                                                  haplotype_dict=haplo_data)

            link_x_s = np.asarray(link_x_s)
            all_n_f[key].append(link_x_s)
            all_n_f_distance[key].append([abs(chrom_pos_focus - p) for p in neighbor_pos])
            linkage_trust_dict[key].append(linkage_trust)
        afc_filter[key] = mean_AFC[key][eval_positions]
        sf_filter[key] = sf[key][eval_positions]
    return all_n_f, all_n_f_distance,afc_filter, all_n_afc,linkage_trust_dict,sf_filter



def calculate_linkage_metric1(selection_positions, positions, eval_positions, haplo_data,afc,afc_all,window=50):
    tmp_haplo_positions=list(sorted(list(haplo_data.keys())))
    # here we consider NO distance values

    linkage_metric=[]
    target_positions=list(selection_positions.keys())
    all_n_f=[]
    all_n_f_rsquare=[]
    all_n_f_D_normal=[]
    all_n_t=[]
    all_n_t_distance=[]
    all_n_f_distance=[]
    all_f_t_distance=[]
    all_f_t= []
    all_n_t_effect=[]
    all_n_afc=[]

    gen_pos_to_arr_pos=dict()
    for arr_pos,gen_pos in enumerate(positions):
        gen_pos_to_arr_pos[gen_pos]=arr_pos

    tmp_linkage=[]
    tmp_afc=[]
    for p_idx,pos in enumerate(eval_positions):



        chrom_pos_focus=positions[pos]
        neighbor_pos=positions[pos-window:pos]+positions[pos+1:pos+window+1]
        # for all neighbors - calculate linkage between all targets
        neighbor_linkages=[]
        neighbor_linkages_tmp=[]
        positions_tmp=[]
        selection_tmp=[]
        neighbor_afc=[]

        focus_afc=afc_all[pos]

        for n in neighbor_pos:
            target_linkage,_,_,filter_position=calculate_all_linkages(n, target_positions, haplotype_dict=haplo_data)
            targets_mean=sum(target_linkage)/len(target_linkage)
            neighbor_linkages.append(targets_mean)
            neighbor_linkages_tmp.append(target_linkage)
            positions_tmp.append([abs(p-n) for p in filter_position])
            selection_tmp.append([selection_positions[p] for p in filter_position])
            neighbor_afc.append(afc_all[gen_pos_to_arr_pos[n]])
            ########################################################
            # For debugging:
            neighbor_afc_tmp=afc_all[gen_pos_to_arr_pos[n]]
            tmp_afc.append((focus_afc,neighbor_afc_tmp))









        all_n_afc.append(neighbor_afc)
        all_n_t.append(neighbor_linkages_tmp)
        all_n_t_distance.append(positions_tmp)
        all_n_t_effect.append(selection_tmp)
        link_s_t=np.asarray(neighbor_linkages)
        #print('s_t',neighbor_linkages)
        #link_s_t=np.mean(neighbor_linkages,axis=-1)

        # for all neighbors - calculate linkage with focus snp
        link_x_s,r_square,normal_D,filter_position=calculate_all_linkages(chrom_pos_focus, neighbor_pos, haplotype_dict=haplo_data)
        tmp_linkage.extend(link_x_s)

        #print('x_s',link_x_s)
        link_x_s=np.asarray(link_x_s)
        all_n_f.append(link_x_s)
        all_n_f_rsquare.append(np.asarray(r_square))
        all_n_f_D_normal.append(np.asarray(normal_D))

        all_n_f_distance.append([abs(chrom_pos_focus-p) for p in filter_position])

        # calculate linkage with focus snp to all targets
        link_f_t,_,_,filter_position_f_t=calculate_all_linkages(chrom_pos_focus, target_positions, haplotype_dict=haplo_data)
        all_f_t.append(link_f_t)

        all_f_t_distance.append([abs(p-chrom_pos_focus) for p in filter_position_f_t])


        # calculate Kramer metric (naive case):
        naive_metric=link_x_s+(link_x_s*link_s_t)
        #print('metric',naive_metric)
        naive_metric=np.mean(naive_metric)
        linkage_metric.append(naive_metric)

    # save everything in pkl_file:
    pkl_data=(eval_positions,all_n_f,all_n_t,all_n_t_distance,all_n_f_distance,all_n_t_effect,afc,all_f_t,all_f_t_distance,all_n_afc,all_n_f_rsquare,all_n_f_D_normal)

    #pkl.dump(pkl_data,open(out_path,'wb'))
    #print('Finish')
    return pkl_data



def calculate_linkage_metric(selection_file_name,eval_positions,haplotype_file,pkl_file):

  # read selected positions:
  selection_positions = read_selection_positions(selection_file_name)

  # read mean AFC out of Pkl file:
  mean_AFC,positions,nucleotide_idx=read_afc(pkl_file)
  positions=positions.tolist()
  # read eval file to get eval positions:
  eval_positions=eval_positions['selected']+eval_positions['unselected']

  # create differnt linkage definitions  and plot correlation with AFC

  # 1. AFC of eval positions:
  afc=mean_AFC[eval_positions]
  # read haplotypes:
  haplo_data = read_haplo_file(haplotype_file,nucleotide_idx)

  return calculate_linkage_metric1(selection_positions,positions,eval_positions,haplo_data,afc,afc_all=mean_AFC)


import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import gzip
import random
import pickle
import numpy as np
rng = np.random.default_rng(seed=0)
dtype=np.float32
tf_dtype= tf.dtypes.float32

base_to_idx={'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}
max_distance=1500
def get_fine_tuning_indices(allele_data, percentag):
    all_keys=[]
    all_indices=[]
    all_af=[]

    for key in allele_data.keys():
        af=np.mean(np.abs(allele_data[key][:,:,0]-allele_data[key][:,:,-1]),axis=-1)
        all_af.extend(af.tolist())
        all_indices.extend(list(range(af.shape[0])))
        all_keys.extend([key]*af.shape[0])

    sorted_af=sorted(zip(all_keys,all_indices,all_af),reverse=True,key=lambda x:x[2])
    #indices=zip(*sorted_af[:int(af.shape[0]*percentag)])
    result=dict()
    for key, ind ,_ in sorted_af[:int(len(all_af)*percentag)]:
        if key not in result:
            result[key]=[]
        result[key].append(ind)

    return result


def read_sync_file_bi_allelic(sync_file, num_generations, num_replicates, num_train_generations,
                              num_train_reps,out_file,haplo_bases=[],zip=True,transpose=False,filter_afc=0):

    #sync_file = open(sync_file, 'r')
    allele_data=dict()
    distance_data=dict()
    nucleotide_idx=dict()
    #start_dist=0
    #seqid='None'
    if zip:
        open_method,open_str=gzip.open,'rb'
    else:
        open_method, open_str = open, 'r'

    with open_method(sync_file, open_str) as f:  # gzip.open(input_file_name, 'rb')
        for line_num,line in enumerate(f):
            if zip:
                line = line.decode('utf-8')
            snp_line=np.zeros((num_train_reps,num_train_generations,4))
            splitted = line.replace("\n", "").split("\t")
            seqid, position, base = splitted[:3]
            print('base',base)
            allele_pair = haplo_bases[line_num][0].split('/')
            print(allele_pair)
            alignment = splitted[3:]
            all_sums=False
            if seqid not in allele_data:
                allele_data[seqid]=[]
                distance_data[seqid]=[]
                nucleotide_idx[seqid]=[]
            #base_idx_1 = base_to_idx[base]
            #some_idx = None
            for pool_num, pool in enumerate(alignment):
                rep_idx=pool_num//num_generations
                gen_idx=pool_num%num_generations

                if rep_idx<num_train_reps and gen_idx<num_train_generations:
                    pool_split=pool.split(":")[:4]
                    #sum_=0

                    for n_idx,e in enumerate(pool_split):
                        coverage=float(e)
                        #sum_+=coverage
                        snp_line[rep_idx,gen_idx,n_idx]=coverage

            if np.sum(snp_line):
                # search  1. and 2. highest freq:

                #freqs=snp_line[:,0]/np.sum(snp_line[:,0],axis=-1,keepdims=True)
                #replace nan with zeros:
                #freqs=np.nan_to_num(freqs,nan=0)
                #freqs=np.mean(freqs,axis=0)
                #idx=list(sorted(enumerate(freqs),key=lambda x:x[1],reverse=True))
                #first_idx,second_idx=idx[0][0],idx[1][0]
                first_idx,second_idx=base_to_idx[allele_pair[0]],base_to_idx[allele_pair[1]]

                snp_line=snp_line[:,:,[first_idx,second_idx]]
                freqs = snp_line / np.sum(snp_line, axis=-1,keepdims=True)
                freqs = np.nan_to_num(freqs, nan=0)[:,:,0]


                allele_data[seqid].append(freqs)
                distance_data[seqid].append(int(position))
                nucleotide_idx[seqid].append([first_idx,second_idx])
                start_dist=int(position)
    all_filter_idx = dict()
    for key in allele_data.keys():
        allele_data[key]=np.asarray(allele_data[key])
        distance_data[key] = np.asarray(distance_data[key])
        nucleotide_idx[key] = np.asarray(nucleotide_idx[key])


        if transpose:
            old_shape=allele_data[key].shape
            allele_data[key] = np.reshape(allele_data[key], newshape=(old_shape[0], old_shape[1] * old_shape[2]), order='C')
            allele_data[key] = np.reshape(allele_data[key], newshape=(old_shape[0], old_shape[1] , old_shape[2]), order='F')
        if filter_afc:
            afcs = np.abs(allele_data[key][:, :, 0] - allele_data[key][:, :, 6])
            afc_rep_max = np.max(afcs, axis=1)
            filter_idx= np.argwhere(afc_rep_max > filter_afc)

            allele_data[key]=allele_data[key][filter_idx]
            distance_data[key] = distance_data[key][filter_idx]
            nucleotide_idx[key]=nucleotide_idx[key][filter_idx]
            all_filter_idx[key]=filter_idx



    fine_tuning_indices=get_fine_tuning_indices(allele_data,percentag=0.1)

    data={'alleles':allele_data,'distances':distance_data,
          'fine_tuning_indices':fine_tuning_indices,'nucleotide_idx':nucleotide_idx}

    data['filter_idx']=all_filter_idx
    print(fine_tuning_indices)
    pickle.dump(data,open(out_file,'wb'))


    return allele_data,distance_data




def find_key_pair(keys, b):
    left, right = 0, len(keys) - 1

    while left <= right:
        mid = (left + right) // 2
        if keys[mid][0] <= b < keys[mid][1]:
            return keys[mid],mid
        elif b < keys[mid][0]:
            right = mid - 1
        else:
            left = mid + 1

    return None,None  # If b is not within any range


def generate_noisy_linear_array(n, m, noise_std=0.05, noise_prob=0.1):
    x = np.linspace(0, 1, m)  # x values from 0 to 1
    arr = np.zeros((n, m))  # Initialize the array

    for i in range(n):
        sign = np.random.choice([1, -1])  # Randomly choose increasing or decreasing

        # Choose b in range [0,1] and ensure a is small enough to keep values in [0,1]
        b = np.random.uniform(0, 1)
        max_a = min(b / max(x), (1 - b) / max(x))  # Ensure values remain in [0,1]
        a = np.random.uniform(0, max_a)

        arr[i, :] = b + sign * a * x  # Generate row

    # Introduce Gaussian noise to some entries
    noise_mask = np.random.rand(n, m) < noise_prob  # Select positions to add noise
    noise = np.random.normal(0, noise_std, (n, m)) * noise_mask  # Generate noise only at selected positions
    arr += noise  # Apply noise

    # Ensure values remain in [0,1]
    arr = np.clip(arr, 0, 1)

    return arr

class Dataset_loader_VAE_bi_allelic_debug(tf.data.Dataset):


    def _generator(num_neighbors,num_train_reps, num_train_gens):

        for i in range(10000):
            neighbor_data = generate_noisy_linear_array((num_neighbors * 2) + 1, num_train_gens,noise_std=0.01, noise_prob=1.0)
            snp_data=neighbor_data[num_neighbors]
            linkage=neighbor_data[:,0]
            neighbor_data=np.expand_dims(neighbor_data,axis=-1)

            yield (snp_data, neighbor_data, linkage)

    def __new__(self, sync_file, num_init_color_channels, num_generations, num_replicates,
                num_train_generations, num_train_reps, num_neighbors, fine_tune, max_dist_neighbor=0,
                decimals=3, seed=3):

        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf_dtype, tf_dtype, tf_dtype),
            output_shapes=((num_train_generations),
                           ((num_neighbors * 2) + 1, num_train_generations, 1), (num_neighbors * 2 + 1)),



            args = (num_neighbors,num_train_reps, num_train_generations)

        )
class Dataset_loader_VAE_bi_allelic(tf.data.Dataset):

    def _generator(num_neighbors, max_dist_neighbor, all_allele_data, all_distance_data, num_train_reps, num_train_gens,
                   all_fine_tuning_indices, fine_tune,all_keys,all_keys_fine_tune):

        # TODO integrate all_keys_fine_tune)
        if fine_tune:
            snp_arr = list(range(all_fine_tuning_indices))
        else:
            num_all_snps = all_allele_data.shape[0]
            snp_arr = list(range(num_all_snps))

        random.shuffle(snp_arr)
        for snp_idx in snp_arr:

            if len(all_keys):
                # 1. find chromosome
                if fine_tune:
                    keys,key_idx=find_key_pair(all_keys_fine_tune,snp_idx)
                    keys=all_keys[key_idx]
                    snp_idx=all_fine_tuning_indices[snp_idx]

                else:
                    keys,key_idx=find_key_pair(all_keys,snp_idx)
                    snp_idx=snp_idx-keys[0]
                    #keys = all_keys[key]
                # 2. show chrom data
                allele_data=all_allele_data[keys[0]:keys[1]]
                distance_data=all_distance_data[keys[0]:keys[1]]
                num_snps = allele_data.shape[0]

            else:
                allele_data=all_allele_data
                distance_data=all_distance_data
                num_snps = allele_data.shape[0]
                if fine_tune:
                    snp_idx=all_fine_tuning_indices[snp_idx]


            snp_row=allele_data[snp_idx]
            rep_idx = random.randrange(num_train_reps)
            snp_data = snp_row[rep_idx,:num_train_gens]
            if snp_idx - num_neighbors < 0:
                neighbor_data = np.zeros(((num_neighbors * 2) + 1, num_train_gens))
                neigbor_distances = np.zeros((num_neighbors * 2)+1)
                neighbor_data_part = allele_data[:snp_idx + num_neighbors + 1, rep_idx,:num_train_gens]
                #print(neighbor_data_part.shape,neighbor_data[num_neighbors - snp_idx:].shape , num_train_gens)
                neighbor_data[num_neighbors - snp_idx:] = neighbor_data_part
                distance_data_part = distance_data[:snp_idx + num_neighbors+1]
                abs_distance = np.abs(distance_data[snp_idx] - distance_data_part)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                neigbor_distances[num_neighbors - snp_idx:] = weight
            elif snp_idx + num_neighbors >= num_snps:
                neighbor_data = np.zeros(((num_neighbors * 2) + 1, num_train_gens))
                neigbor_distances = np.zeros((num_neighbors * 2+1))
                neighbor_data_part = allele_data[snp_idx - num_neighbors:, rep_idx,:num_train_gens]
                neighbor_data[:-(snp_idx + num_neighbors - num_snps + 1)] = neighbor_data_part
                distance_data_part = distance_data[snp_idx - num_neighbors:]
                abs_distance = np.abs(distance_data[snp_idx] - distance_data_part)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                #if snp_idx + num_neighbors - num_snps:
                neigbor_distances[:-(snp_idx + num_neighbors - num_snps+1)] = weight
                #else:
                #    neigbor_distances = distance_data_part
            else:
                neigbor_distances = distance_data[snp_idx - num_neighbors:snp_idx + num_neighbors+1]
                abs_distance = np.abs(distance_data[snp_idx] - neigbor_distances)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                neigbor_distances=weight


                neighbor_data = allele_data[snp_idx - num_neighbors:snp_idx + num_neighbors + 1, rep_idx,:num_train_gens]
            neighbor_data=np.expand_dims(neighbor_data,axis=-1)


            yield (snp_data,neighbor_data,neigbor_distances)

    def __new__(self, sync_file, num_init_color_channels, num_generations, num_replicates,
                num_train_generations, num_train_reps,num_neighbors,fine_tune,max_dist_neighbor=0,
                decimals=3, seed=3):
        max_dist_neighbor=0
        np.random.seed(seed)
        random.seed(seed)
        #Bi_Allelic reader:
        data=pickle.load(open(sync_file,'rb'))
        allele_data,distance_data,fine_tuning_indices=data['alleles'],data['distances'],data['fine_tuning_indices']

        #allele_data, distance_data, fine_tuning_indices = data['alleles'][], data['distances'], data['fine_tuning_indices']
        #distance_data=distance_data[1:]-distance_data[:-1]
        print(sync_file)
        #print(allele_data.shape)
        #print(distance_data.shape)
        #allele_data=np.arcsin(np.sqrt(allele_data))/np.arcsin(np.sqrt(1))


        all_allele_data,all_distance_data,all_fine_tuning_indices=[],[],[]
        all_keys=[]
        all_keys_fine_tune=[]
        lens=0
        lens_fine_tune=0
        if isinstance(allele_data,dict):
            for key in allele_data.keys():
                if allele_data[key].shape[0]:
                    all_keys.append([lens,lens+allele_data[key].shape[0]])
                    lens+=allele_data[key].shape[0]

                    all_keys_fine_tune.append([lens_fine_tune,lens_fine_tune+len(fine_tuning_indices[key])])
                    lens_fine_tune+=len(fine_tuning_indices[key])

                    all_allele_data.extend(np.squeeze(allele_data[key]))
                    all_distance_data.extend(np.squeeze(distance_data[key]))
                    #all_fine_tuning_indices.extend(len(all_fine_tuning_indices)+np.squeeze(np.asarray(fine_tuning_indices[key])))
                    all_fine_tuning_indices.extend(np.squeeze(np.asarray(fine_tuning_indices[key])))
                else:
                    print('DEBUg key',key,'has no SNPs')
        else:
            all_allele_data=allele_data
            all_distance_data=distance_data
            all_fine_tuning_indices=fine_tuning_indices
        all_allele_data=np.asarray(all_allele_data)
        print('shape allele data',all_allele_data.shape,flush=True)
        print('keys',all_keys,all_keys_fine_tune)


        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf_dtype, tf_dtype, tf_dtype),
            output_shapes=((num_train_generations),
                           ((num_neighbors*2)+1,num_train_generations,1), (num_neighbors*2+1)),

            args=(num_neighbors,max_dist_neighbor,all_allele_data,all_distance_data,
                  num_train_reps,num_train_generations,all_fine_tuning_indices,fine_tune,all_keys,all_keys_fine_tune)
        )

class DS_loader_bi_allelic_replicates(tf.data.Dataset):

    def _generator(num_neighbors,max_dist_neighbor,allele_data,distance_data,num_train_reps,num_train_gens,fine_tuning_indices,fine_tune):


        num_snps=allele_data.shape[0]
        if fine_tune:
            snp_arr=fine_tuning_indices
        else:
            snp_arr=list(range(num_snps))
        random.shuffle(snp_arr)
        for snp_idx in snp_arr:
            snp_row=allele_data[snp_idx]
            snp_data = snp_row[:,:num_train_gens]
            if snp_idx - num_neighbors < 0:
                neighbor_data = np.zeros(((num_neighbors * 2) + 1,num_train_reps,num_train_gens))
                neigbor_distances = np.zeros((num_neighbors * 2)+1)
                neighbor_data_part = allele_data[:snp_idx + num_neighbors + 1, :,:num_train_gens]
                #print(neighbor_data_part.shape,neighbor_data[num_neighbors - snp_idx:].shape , num_train_gens)
                neighbor_data[num_neighbors - snp_idx:] = neighbor_data_part
                distance_data_part = distance_data[:snp_idx + num_neighbors+1]
                abs_distance = np.abs(distance_data[snp_idx] - distance_data_part)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                neigbor_distances[num_neighbors - snp_idx:] = weight
            elif snp_idx + num_neighbors >= num_snps:
                neighbor_data = np.zeros(((num_neighbors * 2) + 1,num_train_reps, num_train_gens))
                neigbor_distances = np.zeros((num_neighbors * 2+1))
                neighbor_data_part = allele_data[snp_idx - num_neighbors:, :,:num_train_gens]
                neighbor_data[:-(snp_idx + num_neighbors - num_snps + 1)] = neighbor_data_part
                distance_data_part = distance_data[snp_idx - num_neighbors:]
                abs_distance = np.abs(distance_data[snp_idx] - distance_data_part)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                #if snp_idx + num_neighbors - num_snps:
                neigbor_distances[:-(snp_idx + num_neighbors - num_snps+1)] = weight
                #else:
                #    neigbor_distances = distance_data_part
            else:
                neigbor_distances = distance_data[snp_idx - num_neighbors:snp_idx + num_neighbors+1]
                abs_distance = np.abs(distance_data[snp_idx] - neigbor_distances)
                weight=(max_distance - abs_distance) / max_distance
                weight[weight < 0] = 0.0
                neigbor_distances=weight
                neighbor_data = allele_data[snp_idx - num_neighbors:snp_idx + num_neighbors + 1, :, :num_train_gens]


            yield (snp_data,neighbor_data,neigbor_distances)

    def __new__(self, sync_file, num_init_color_channels, num_generations, num_replicates,
                num_train_generations, num_train_reps,num_neighbors,fine_tune,max_dist_neighbor=0,
                decimals=3, seed=3):
        max_dist_neighbor=0
        np.random.seed(seed)
        random.seed(seed)
        #Bi_Allelic reader:
        data=pickle.load(open(sync_file,'rb'))
        allele_data,distance_data,fine_tuning_indices=data['alleles'],data['distances'],data['fine_tuning_indices']
        #distance_data=distance_data[1:]-distance_data[:-1]
        print(sync_file)
        #print(allele_data.shape)
        #print(distance_data.shape)
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf_dtype, tf_dtype, tf.dtypes.int32),
            output_shapes=((num_train_reps,num_train_generations),
                           ((num_neighbors*2)+1,num_train_reps,num_train_generations), (num_neighbors*2+1)),

            args=(num_neighbors,max_dist_neighbor,allele_data,distance_data,num_train_reps,num_train_generations,fine_tuning_indices,fine_tune)
        )

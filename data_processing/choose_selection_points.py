import random

random.seed(0)
def get_positions(haplo_file):
    positions=dict()
    haplo_in_stream = open(haplo_file,'r')
    for line in haplo_in_stream:
        splitted=line.split('\t')
        chrom,position,nukleotide,alleles=splitted[:4]
        haplos=splitted[4].replace('\n','').replace(' ','')
        freq=haplos.count(nukleotide)/len(haplos)

        if chrom not in positions:
            positions[chrom]=[]
        positions[chrom].append((int(position),nukleotide,alleles,freq))
    haplo_in_stream.close()
    return positions


def declare_targets(org_haplo_file,target_file_out,number_of_targets,add_effect_distance=0.2,min_freq=0.45,max_freq=0.55):

    out_stream=open(target_file_out,'w')
    positions=get_positions(org_haplo_file)
    for chrom, pos_data in positions.items():
        # colect start and end distance
        start,end=pos_data[0][0],pos_data[-1][0]
        len_region=end-start
        # create snp regions
        bin_size=len_region//number_of_targets

        additive_effects=[add_effect_distance*i for i in range(1,number_of_targets+1)]
        random.shuffle(additive_effects)
        for target_idx in range(number_of_targets):
            bin_start=start+bin_size*target_idx
            bin_end=start+bin_size*(target_idx+1)
            # 2. filter positions according to freq
            filter_pos=[]
            for p_d in pos_data:
                if bin_start<=p_d[0]<=bin_end and min_freq<=p_d[-1]<=max_freq:
                    filter_pos.append(p_d)
            # choose positions
            target=random.choice(filter_pos)
            # choose allele order
            target_split=target[2].split('/')
            random.shuffle(target_split)
            new_line = chrom + '\t' + str(target[0]) + '\t' + target_split[0]+'/'+target_split[1] + '\t' + str(additive_effects[target_idx]) + '\t0\n'
            out_stream.write(new_line)

    out_stream.close()


import numpy as np
import pickle as pkl
from plot_functions import plot_dataset_details_box_chroms
from variables import fix_variables
from linkage.LD_calculator import calculate_LD_on_subset_chrom
from data_processing.file_reader import read_estimated_s_positions, read_targets_acc_to_afc


def read_all_data(animal,chromosomes, work_from='work_lap',ld_idx=3):
    AFC = dict()
    estimated_s = dict()
    Nes = dict()

    # 1. get direct neighbor linkage
    # 2. get random neighbor linkage

    LD = calculate_LD_on_subset_chrom(animal, work_from=work_from, sample_size=1000, n_window=50,ld_idx=ld_idx,chroms=chromosomes)

    sync_file = fix_variables[work_from][animal]["sync_process"]
    eval_path = fix_variables[work_from][animal]['eval_positions']
    targets_allchrom, no_targets_all_chroms, targets_org_pos_all_chroms, no_targets_org_pos_all_chroms = read_targets_acc_to_afc(sync_file,eval_path)
    data = pkl.load(open(sync_file, 'rb'))



    for chrom  in chromosomes:

        AFC_partial = []
        estimated_s_partial = []
        ne_gt = fix_variables[animal]['Nes'][chrom]
        Nes[chrom]=ne_gt

        allele_data = np.squeeze(data['alleles'][chrom])
        targets, no_targets, targets_org_pos, no_targets_org_pos=targets_allchrom[chrom], no_targets_all_chroms[chrom], targets_org_pos_all_chroms[chrom], no_targets_org_pos_all_chroms[chrom]
        print(allele_data.shape)
        print(targets)
        AFC_partial.extend([np.mean(np.abs(allele_data[no_targets, :, 0] - allele_data[no_targets, :, -1]), axis=1),
                            np.mean(np.abs(allele_data[targets, :, 0] - allele_data[targets, :, -1]), axis=1)])


        estimated_s_file = fix_variables[work_from][animal]["estimated_s"][0][0]
        estimated_s_partial.extend(
            read_estimated_s_positions(estimated_s_file, [no_targets_org_pos, targets_org_pos],chrom=chrom))

        AFC[chrom]=AFC_partial
        estimated_s[chrom]=estimated_s_partial

    return  AFC, LD, estimated_s, Nes
def get_SF_data(animal,chromosomes, work_from='work_lap',ld_idx=1):
    SF =dict()
    sync_file = fix_variables[work_from][animal]["sync_process"]
    eval_path = fix_variables[work_from][animal]['eval_positions']
    targets_allchrom, no_targets_all_chroms, targets_org_pos_all_chroms, no_targets_org_pos_all_chroms = read_targets_acc_to_afc(sync_file,eval_path)

    data = pkl.load(open(sync_file, 'rb'))

    for chrom in chromosomes:
        SF_partial = []
        allele_data = np.squeeze(data['alleles'][chrom])

        targets, no_targets, targets_org_pos, no_targets_org_pos=targets_allchrom[chrom], no_targets_all_chroms[chrom], targets_org_pos_all_chroms[chrom], no_targets_org_pos_all_chroms[chrom]

        # shape: (# snps, rep, gen)
        SF_partial.extend(
            [np.mean(allele_data[no_targets, :, 0], axis=1), np.mean(allele_data[targets, :, 0], axis=1)])
        SF[chrom]=SF_partial
    return SF



if __name__ == '__main__':

    work_from='home'
    animal='experimental'
    meta_data_file_name='meta_data_plus_LD'+animal+'.pkl'

    use_abs_vals=False

    x_axis_names=['2L','2R','3L','3R','4','X']

    # show ground truth data
    # s + chm on the estimated datasets
    chroms = ['2L', '2R', '3L', '3R', '4', 'X']
    AFC,LD,estimated_s,Nes=read_all_data(animal,work_from=work_from,ld_idx=2,chromosomes=chroms)
    meta_data_file_name='experimental_new_s.pkl'
    pkl.dump([AFC,LD,estimated_s,Nes],open(meta_data_file_name,'wb'))

    SF=get_SF_data('experimental_flip',work_from=work_from,ld_idx=2,chromosomes=chroms)
    pkl.dump(SF, open('empirical_flip_SF_new.pkl', 'wb'))
    #print('Save SF',flush=True)

    SF=pkl.load(open('empirical_flip_SF.pkl', 'rb'))

    # plot eveything:
    data_subtitles = [
        ('SF $f^{(0)}$', ['Low AFC', 'High AFC'],'violin'),#violin
        ('LD $r^2$', ['$\mathcal{N}_{50}$'],'violin'), #violin # ['$\mathcal{N}_2$','$\mathcal{N}_{50}$']
        #('AFC', ['low AFC', 'high AFC'],''),
        ('Estimated $N_{e}$', [''],''),
        ('Estimated $|s|$', ['Low AFC', 'High AFC'],''),
    ]
    [AFC, _, estimated_s, Nes] = pkl.load(open('experimental_new_s.pkl', 'rb'))
    [AFC,LD,_,Nes] = pkl.load(open('meta_data_plus_LDexperimental.pkl','rb'))
    datas=[SF,LD,Nes,estimated_s]#AFC,
    plot_dataset_details_box_chroms(datas,data_subtitles,plt_out_name='experimental_data_description')





    

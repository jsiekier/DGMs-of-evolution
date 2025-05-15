import numpy as np
import pickle
from plot_scripts.plot_functions import  plot_dataset_details_box
from variables import fix_variables
from linkage.LD_calculator import calculate_LD_on_subset
from data_processing.file_reader import  read_targets,     read_estimated_s_positions


def read_all_data(animals, work_from='work_lap'):
    #SF = []
    AFC = []
    estimated_s = []
    #log_p = []
    Nes = []

    # 1. get direct neighbor linkage
    # 2. get random neighbor linkage

    LD = calculate_LD_on_subset(animals, work_from=work_from, sample_size=1000, n_window=50)

    for animal in animals:
        #animal_gt = animal_stuff[0][0]
        #SF_partial = []
        AFC_partial = []
        estimated_s_partial = []
        ne_gt = fix_variables[animal]['Nes']
        Nes.append([ne_gt])
        #log_p_partial = []
        #for animal_noise in animal_stuff[0][1:]:

        selection_pos_file = fix_variables[work_from][animal]["targets"]
        sync_file = fix_variables[work_from][animal]["sync_process"]
        targets, no_targets, targets_org_pos, no_targets_org_pos = read_targets(selection_pos_file, sync_file)

        data = pickle.load(open(sync_file, 'rb'))
        if isinstance(data['alleles'], dict):
            allele_data = data['alleles']['2L']
        else:
            allele_data = data['alleles']
        AFC_partial.extend([np.mean(np.abs(allele_data[no_targets, :, 0] - allele_data[no_targets, :, -1]), axis=1),
                            np.mean(np.abs(allele_data[targets, :, 0] - allele_data[targets, :, -1]), axis=1)])



        estimated_s_file = fix_variables[work_from][animal]["estimated_s"][0][0]
        estimated_s_partial.extend(
            read_estimated_s_positions(estimated_s_file, [no_targets_org_pos, targets_org_pos]))

            #log_p_partial.extend([[cmh_dict[p] for p in no_targets_org_pos], [cmh_dict[p] for p in targets_org_pos]])

        #SF.append(SF_partial)
        AFC.append(AFC_partial)
        estimated_s.append(estimated_s_partial)
        #log_p.append(log_p_partial)

    return  AFC, LD, estimated_s, Nes


if __name__ == '__main__':

    work_from='mogon'#'work_lap'#
    experiment_number=1
    meta_data_file_name='meta_data_plus_exp2.pkl'


    x_axis_names = [["$\mathcal{N}(.9,.5)$\n$p_{LD}=.0$","$\mathcal{N}(.8,.5)$\n$p_{LD}=.0$","$\mathcal{N}(.7,.5)$\n$p_{LD}=.0$",
                    "$\mathcal{N}(.9,.5)$\n$p_{LD}=.2$","$\mathcal{N}(.8,.5)$\n$p_{LD}=.2$","$\mathcal{N}(.7,.5)$\n$p_{LD}=.2$"],
                    #["$n_{LD}=.00;|T|=50$","$n_{LD}=.00;|T|=25$","$n_{LD}=.00;|T|=10$",
                    # "$n_{LD}=.04;|T|=50$","$n_{LD}=.04;|T|=25$","$n_{LD}=.04;|T|=10$",
                    # "$n_{LD}=.08;|T|=50$","$n_{LD}=.08;|T|=25$","$n_{LD}=.08;|T|=10$",
                    # "$n_{LD}=.12;|T|=50$","$n_{LD}=.12;|T|=25$","$n_{LD}=.12;|T|=10$",]
    ]
    x_axis_name=[]
    for n_ld in ['.00','.04','.08','.12']:
        for num_t in ['50','25','10']:
            x_axis_name.append("$n_{LD}="+n_ld+"$\n$|T|="+num_t+"$")
    x_axis_names.append(x_axis_name)
    x_axis_names=x_axis_names[experiment_number]
    animals=[["gauss_09_05_noreg_pw0_1000_a01_noise14","gauss_08_05_noreg_pw0_1000_a01_noise14","gauss_07_05_noreg_pw0_1000_a01_noise14",
              "gauss_09_05_noreg_pw20_1000_a01_noise14","gauss_08_05_noreg_pw20_1000_a01_noise14","gauss_07_05_noreg_pw20_1000_a01_noise14"],
             ['max_0_r_t_50_n_noise14','max_0_r_t_25_n_noise14','max_0_r_t_10_n_noise14',
              'max_4_r_t_50_n_noise14','max_4_r_t_25_n_noise14','max_4_r_t_10_n_noise14',
              'max_8_r_t_50_n_noise14','max_8_r_t_25_n_noise14','max_8_r_t_10_n_noise14',
              'max_12_r_t_50_n_noise14','max_12_r_t_25_n_noise14','max_12_r_t_10_n_noise14',]][experiment_number]



    # show ground truth data
    # s + chm on the estimated datasets
    AFC,LD,estimated_s,Nes=read_all_data(animals,work_from=work_from)

    pickle.dump([AFC,LD,estimated_s,Nes],open(meta_data_file_name,'wb'))

    # plot eveything:

    [AFC,LD,estimated_s,Nes] = pickle.load(open(meta_data_file_name,'rb'))
    data_subtitles=[
                    ('LD', ['$\mathcal{N}_{50}$']),#'$\mathcal{N}_2$',
                    ('AFC',['No targets','Targets']),
                    ('Estimated $N_{e}$', ['']),
                    ('Estimated $|s|$',['No targets','Targets']),
    ]
    datas=[LD,AFC,Nes,estimated_s]
    plot_dataset_details_box(datas,data_subtitles,x_axis_names,plt_out_name='dataset_summary_plus_exp2')
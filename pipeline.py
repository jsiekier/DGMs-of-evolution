import argparse
from vae.train_snp_vae import main_training
import tensorflow as tf
print('GPU:',tf.config.list_physical_devices('GPU'))


def parse_args():
    notes='small gene enc'
    animal='uniform_01_pw0'#'uniform_01_pw0' ##'10_target_link_uniform'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', default='gpu', type=str)
    argparser.add_argument('--item_assignment', default=1, type=int)
    argparser.add_argument('--repeat_iterations', default=5, type=int)#2#


    argparser.add_argument('--non_linearity', default='tanh', type=str)#tanh
    argparser.add_argument('--batch_size', default=20, type=int)
    argparser.add_argument('--num_gene_batches', default=1, type=int)#1
    argparser.add_argument('--work_from', default="work_lap", type=str)
    argparser.add_argument('--animal', default=animal, type=str)
    argparser.add_argument('--num_init_color_chanels', default=4, type=int)
    argparser.add_argument('--num_layers', default=6, type=int)
    argparser.add_argument('--train_sessions', default=30000, type=int)#300#10000
    argparser.add_argument('--pretrain_sessions', default=0, type=int)
    argparser.add_argument('--end_z_dim', default=10, type=int)
    argparser.add_argument('--reduce_shortcut', default=0, type=int)#TODO


    #tanh_l1_nofinetune_noBN_z10_l4_300
    argparser.add_argument('--print_every', default=1, type=int)
    argparser.add_argument('--nn_name', default='t', type=str)#just_posinfo- interpretable2525_multiply#le01_l3_interp_mul_b_i_beta_20
    argparser.add_argument('--dilations', default=2, type=int)
    argparser.add_argument('--dropout', default=0.0, type=float)
    argparser.add_argument('--filter_size', default=9, type=int)#9
    argparser.add_argument('--batchnorm_axis', default=0, type=int)
    argparser.add_argument('--batchnorm_renorm', default=0, type=int)
    argparser.add_argument('--round_decimal', default=7, type=int)
    argparser.add_argument('--use_map', default=2, type=int)
    argparser.add_argument('--use_resnet', default=1, type=int)
    argparser.add_argument('--gauss_lay', default=[], type=int, nargs='+')
    argparser.add_argument('--gauss_std', default=0.0, type=float)
    argparser.add_argument('--learning_rate', default=0.001, type=float)
    argparser.add_argument('--opt', default='adam', type=str)#adam
    argparser.add_argument('--nn', default='VAE', type=str)#choices 'E' 'AE' 'VAE'
    argparser.add_argument('--rec_loss', default='l1', type=str)#choices 'l1' - 'entropy'

    argparser.add_argument('--use_selection_coefficient', default=0, type=int)
    argparser.add_argument('--recursive_enc', default=0, type=int)
    argparser.add_argument('--recursive_dec', default=0, type=int)
    argparser.add_argument('--num_snp_layers', default=6, type=int)
    argparser.add_argument('--num_neighbors', default=50, type=int)#125
    argparser.add_argument('--integrate_gene', default=1, type=int)#1
    argparser.add_argument('--position_info', default=0, type=int)
    argparser.add_argument('--layer_dims', default=[100,100,50,50,25,25,25], type=int, nargs='+')#num_train_it
    argparser.add_argument('--num_train_it', default=0, type=int)
    argparser.add_argument('--num_replicates', default=0, type=int)
    argparser.add_argument('--model_num', default='0', type=str)
    argparser.add_argument('--include_parallel_info', default=0, type=int)

    argparser.add_argument('--time_batch_size', default=4, type=int)
    argparser.add_argument('--fine_tune', default=0, type=int)
    argparser.add_argument('--KLD', default=3, type=int)
    argparser.add_argument('--prior', default='nice',type=str)
    argparser.add_argument('--linkage_prior', default=0, type=int)
    argparser.add_argument('--bi_allelic', default=1, type=int)
    argparser.add_argument('--enc_type', default='sim', type=str,choices=['MLP','CNN','sim'])
    argparser.add_argument('--loss_type', default='EMD', type=str, choices=['EMD', 'L2','KDE'])
    argparser.add_argument('--sf_ex_alpha', default=0.0, type=float)
    argparser.add_argument('--exclude_sf', default=0, type=int)
    argparser.add_argument('--KLD_anneal', default=0, type=int)
    argparser.add_argument('--debug', default=1, type=int)
    argparser.add_argument('--multiply_result', default=0, type=int)
    arguments = argparser.parse_args()



    return arguments



if __name__ == '__main__':

    tmp_data_folder='data/'
    args = parse_args()

    kld_influence = 1 / (10 ** args.KLD)  # 0.001 args.KLD #
    num_models=int(args.model_num)

    sample_size=100
    num_unselected_snps=5000
    num_selected_snps_region=4000
    model_type='latest'

    base_model_name=args.nn_name
    plot_name_ = base_model_name+'extended'

    model_names=[]
    use_all_train_data=True
    model_num=num_models
    # train VAE model

    args.fine_tune=0
    VAE_name=base_model_name.replace('AE','VAE')
    args.nn_name = VAE_name+'_'+'KLD'+str(args.KLD) + '_' + str(model_num)
    args.nn='VAE'
    args.train_sessions=1000#1#240#60
    args.print_every=1


    main_training(args,use_tensorboard=False,tmp_data_folder=tmp_data_folder,kld_influence=kld_influence)
    print('break - new model')
    #
    args.learning_rate /= 10
    args.fine_tune=1
    args.train_sessions=80#20
    args.print_every=1

    main_training(args,use_tensorboard=False,tmp_data_folder=tmp_data_folder,kld_influence=kld_influence)




import random
import numpy as np
import warnings
import pickle as pkl
from variables import fix_variables


np.random.seed(0)
random.seed(0)

from Data_loader import get_fine_tuning_indices


def sample_alleles(p, size, mode=("coverage", "individuals"), Ncensus=None, ploidy=2):
    # Determine the number of return values
    maxlen = max(len(p), len(size))

    # Check the length of p, size, and mode parameters
    if maxlen % len(p) != 0 or maxlen % len(size) != 0:
        warnings.warn(
            f"Parameters differ in length and are not multiples of one another: length(p)={len(p)}, length(size)={len(size)}")

    # Sample allele counts according to the specified method
    if mode == "coverage":
        # If length of 'size' equals '1', generate target coverage values using the Poisson distribution,
        # otherwise use values of 'size' directly
        if len(size) == 1:
            cov = np.random.poisson(lam=size[0], size=p.shape)
        else:
            cov = size

        # Sample allele frequencies from the Binomial distribution based on 'cov' and 'p'
        p_smpld = np.random.binomial(n=cov, p=p, size=None) / cov

        # Return results, including coverage values if they were drawn from a Poisson distribution
        #if len(size) == 1:
        #    return pd.DataFrame({"p_smpld": p_smpld, "size": cov})
        #else:
        return p_smpld,cov
    elif mode == "individuals":
        # If length of 'size' is larger than 1, send a warning message
        if len(size) > 1:
            warnings.warn("Only the first element in 'size' will be used, because sampling mode is 'individuals'")

        # Sample random allele frequencies from the Hypergeometric distribution
        new_prod_1=(p * Ncensus * ploidy).astype(int)
        new_prod_2 =((1 - p) * Ncensus * ploidy).astype(int)
        chroms= size[0] * ploidy
        p_smpld = np.random.hypergeometric(new_prod_1,new_prod_2,chroms,size=None).astype(np.float32) #/ (size[0] * ploidy)
        p_smpld=p_smpld*(1/(size[0] * ploidy))
        return p_smpld


def make_noise_file(animal_in,animal_out,N_s1,N_s2,work_from='mogon',Ncensus = 1000,fine_tune_percentage=0.1,chrom=''):

    file_name=fix_variables[work_from][animal_in]['sync_process']#base_path+'linked_gauss_90_50_randD_01_ad.pkl'
    alleles_data=pkl.load(open(file_name,'rb'))
    if chrom!='':
        alleles=alleles_data['alleles'][chrom]
    else:
        alleles=alleles_data['alleles']
    distance_data=alleles_data['distances']

    p = alleles
    ploidy = 2

    allele_individuals = sample_alleles(p, N_s1, "individuals", Ncensus, ploidy)
    allele_coverage,coverage = sample_alleles(allele_individuals, N_s2, "coverage", Ncensus, ploidy)
    allele_coverage={chrom:allele_coverage}
    coverage={chrom:coverage}
    fine_tuning_indices=get_fine_tuning_indices(allele_coverage, percentag=fine_tune_percentage)

    # create new fine tuning index!!
    out_file=fix_variables[work_from][animal_out]['sync_process']#base_path+'linked_gauss_90_50_randD_01_ad_noise_200_200.pkl'
    data = {'alleles': allele_coverage, 'distances': distance_data, 'fine_tuning_indices': fine_tuning_indices,'cov':coverage}
    print(fine_tuning_indices)
    pkl.dump(data, open(out_file, 'wb'))





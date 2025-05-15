import numpy as np

rng = np.random.default_rng(seed=0)

def simulate(start_freq, Ne, tracking_points, num_simulations,s=0,h=0.5,n_round=3,n_sampling=1000,n_census=1000,coverage=None):
    ploidy = 2
    Ne *= ploidy  # diploid org.
    new_freqs=np.asarray([start_freq]*num_simulations)
    tracked_freqs=[np.copy(new_freqs)]

    p_old = new_freqs
    q_old = 1-new_freqs

    p = new_freqs
    q = 1-new_freqs
    wAA = 1+s
    wAa = 1+h*s
    waa = 1


    for g in range(1,tracking_points[-1]+1):

        # compute mean fitness
        w = p *p * wAA + 2 * p * q * wAa + q *q* waa


        # apply selection and random drift
        p = (wAA * p *p + wAa * p * q) / w


        try:
            p = rng.binomial(Ne, p, num_simulations)/Ne #rbinom(length(p), Ne, p) / Ne
        except Exception:
            print('Error',g,p,p_old,q_old,Ne,wAA,waa,wAa,s,new_freqs)
        q = 1 - p

        if g in tracking_points:
            tracked_freqs.append(np.copy(p))

    return tracked_freqs






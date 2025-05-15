import numpy as np
from scipy.special import logit
from scipy.stats import linregress
from scipy.optimize import curve_fit
import statsmodels.api as sm
import warnings

def simulate(start_freq, Ne, tracking_points, num_simulations,s=0,h=0.5,n_round=3,approximate=False,haploid=False):



    # Continuous-time approximation for infinite population size
    if Ne is None and approximate and (haploid or np.all(h == 0.5)):
        #print('test')
        t_all = np.repeat(tracking_points, num_simulations)
        traj = 1 / (1 + ((1 - start_freq) / start_freq) * np.exp(-s * t_all / (1 if haploid else 2)))
        tracked_freqs = traj.reshape(num_simulations, len(tracking_points))
    else:
        Ne *= 2  # diploid org.
        #start_freq=1.0
        new_freqs=np.asarray([start_freq]*num_simulations)
        new_freqs = np.round(new_freqs, decimals=n_round)
        tracked_freqs=[np.copy(new_freqs)]

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

            p =   rng.binomial(Ne, p, num_simulations)/Ne #rbinom(length(p), Ne, p) / Ne
            q = 1 - p

            # if necessary then save current allele frequency to results
            if g in tracking_points:
                tracked_freqs.append(np.copy(p))
    #print(np.asarray(tracked_freqs).shape)
    return tracked_freqs

def scale_af(af, method="logit"):
    if method == "logit":
        # Clip allele frequencies to avoid logit issues (logit undefined for 0 or 1)
        af = np.clip(af, 1e-9, 1 - 1e-9)
        return logit(af)
    else:
        raise ValueError("Unsupported scaling method.")

##This function fits a non-linear least squares (NLS) model to estimate the selection coefficient (s) and,
# optionally, the dominance coefficient (h) using a Wright-Fisher trajectory (wf.traj).
# If the estimation process encounters issues, it attempts to handle errors gracefully.
def nls_sh(ctraj, Ne, h, haploid, s_start=0.1, h_start=0.5, approximate=False):
    af = np.asarray(ctraj['af'])
    t = np.asarray(ctraj['t'])

    # Initial result in case of failure
    res = {'s': np.nan, 'h': np.nan if np.isnan(h) else h}

    # Define limits for parameter estimates
    s_lim = [-np.inf, np.inf]
    h_lim = [-np.inf, np.inf]

    # Define the model function for curve fitting

    def model_fixed_h(t, sEst):
        wf_simu=simulate(start_freq=af[0], Ne=None, tracking_points=t, s=sEst, h=h,num_simulations=1,haploid=haploid,approximate=approximate)#, haploid=haploid, approximate=approximate)
        return np.asarray(wf_simu)[0]

    try:
        # Fit the model using non-linear least squares

        # Estimate only s (h is fixed)
        popt, _ = curve_fit(model_fixed_h, t, af, p0=[s_start], maxfev=10000)
        s_est = popt[0]
        h_est = h

        # Check if estimates are within limits
        s_fix = None if s_lim[0] <= s_est <= s_lim[1] else s_lim[0] if s_est < s_lim[0] else s_lim[1]
        h_fix = None if not np.isnan(h) or (h_lim[0] <= h_est <= h_lim[1]) else h_lim[0] if h_est < h_lim[0] else h_lim[1]


        # Assign the estimates
        res['s'] = s_fix if s_fix is not None else s_est
        res['h'] = h_fix if h_fix is not None else h_est

    except Exception as e:
        print(f"Initial fit failed: {e}")

    # Ensure the estimates are within the limits
    if res['s'] < s_lim[0]:
        res['s'] = s_lim[0]
    elif res['s'] > s_lim[1]:
        res['s'] = s_lim[1]

    if 'h' in res and not np.isnan(res['h']):
        if res['h'] < h_lim[0]:
            res['h'] = h_lim[0]
        elif res['h'] > h_lim[1]:
            res['h'] = h_lim[1]

    return res


def lm_s(ctraj, haploid, maxiter=10, tol=0.01):
    # Validate ctraj input
    if not isinstance(ctraj, dict) or 'af' not in ctraj or 't' not in ctraj:
        raise ValueError("The parameter 'ctraj' is not specified properly. It must be generated with consensus_traj and contain 'af' and 't'.")

    af = np.asarray(ctraj['af'])
    t = np.asarray(ctraj['t'])

    if len(af) == 0 or len(t) == 0:
        raise ValueError("The allele frequency ('af') and time ('t') must have non-zero length.")

    # Remove uninformative time points
    af_diff = np.abs(np.diff(af))
    max_diff = np.max(af_diff)
    # Identify consecutive points with negligible difference
    mask = af_diff < max_diff * 0.01
    if np.any(mask):
        remove_indices = np.arange(len(af) - len(mask), len(af))[mask]
        af = np.delete(af, remove_indices)
        t = np.delete(t, remove_indices)

    # Filter out allele frequencies that are exactly 0 or 1
    ikeep = np.where((af != 0) & (af != 1))[0]

    # If all time points were filtered out, return NA values
    if len(ikeep) < 2:
        return {'s': np.nan, 'p0': np.nan}

    af = af[ikeep]
    t = t[ikeep]

    # Scale allele frequencies using logit transformation
    scaled_af = scale_af(af)

    # Fit a linear model: scaled_af ~ t
    slope, intercept, _, _, _ = linregress(t, scaled_af)
    s = slope * (2 if not haploid else 1)
    p0 = 1 / (1 + np.exp(-intercept))

    # Bias correction for large s values
    if s != 0 and p0 != 0 and p0 != 1 and maxiter >= 1:
        i = 0
        s_delta = 0
        s_est = np.nan

        while True:
            # Simulate trajectory using the current estimates of s and p0
            trj_sim = simulate(start_freq=p0, Ne=None, tracking_points=t, s=s + s_delta,num_simulations=1)#, haploid=haploid)

            # Check if polymorphic at fewer than 2 time points
            if np.sum((trj_sim > 0) & (trj_sim < 1)) < 2:
                warnings.warn("Correction unsuccessful. Selection estimates may be inaccurate.")
                break

            # Re-estimate parameters from the simulated trajectory
            fit_slope, fit_intercept, _, _, _ = linregress(t, scale_af(trj_sim))
            s_est = fit_slope * (2 if not haploid else 1)

            # Update s_delta
            s_delta = s + s_delta - s_est
            i += 1

            # Check for convergence or maximum iterations
            if i >= maxiter or np.isnan(s_est) or abs((s_est - s) / s) <= tol:
                break

        if i >= maxiter and abs((s_est - s) / s) > tol:
            warnings.warn("Max iterations reached without convergence. Selection estimates may be inaccurate.")

        # Correct the selection coefficient
        s += s_delta

    return {'s': s, 'p0': p0}



def fit_quadratic_model(af, t):
    """Fit a quadratic model to the scaled allele frequencies."""
    af_scaled = scale_af(af)

    # Create polynomial features (t and t^2)
    X = np.column_stack([np.ones_like(t), t, t**2])

    # Fit the linear model
    model = sm.OLS(af_scaled, X).fit()

    # Extract the p-value for the quadratic term (coefficient of t^2)
    p_value = model.pvalues[2]  # Index 2 corresponds to the t^2 term

    return model, p_value
#This function determines whether the Linear Least Squares (LLS) method can be used for estimation,
# based on a fitted polynomial model and a p-value threshold.
def use_lls(ctraj, h, haploid, p_min=0.10):
    # Check for haploid or dominance coefficient h == 0.5
    if not haploid and not np.isnan(h) and h != 0.5:
        return False

    # Validate ctraj input
    if not isinstance(ctraj, dict) or 'af' not in ctraj or 't' not in ctraj:
        raise ValueError("The parameter 'ctraj' is not specified properly. It must be generated with consensus_traj and contain 'af' and 't'.")

    af = np.asarray(ctraj['af'])
    t = np.asarray(ctraj['t'])

    if len(af) == 0 or len(t) == 0:
        raise ValueError("The allele frequency ('af') and time ('t') must have non-zero length.")

    p_min = float(p_min)
    if not isinstance(p_min, (int, float)):
        raise ValueError("'p_min' must be a numeric value.")

    # Filter out allele frequencies that are exactly 0 or 1
    ikeep = np.where((af != 0) & (af != 1))[0]

    # If fewer than 3 time points remain, use LLS
    if len(ikeep) < 3:
        return True
    model, p_value = fit_quadratic_model(af, t)

    # Scale allele frequencies using logit transformation
    #scaled_af = scale_af(af[ikeep])

    # Fit a quadratic polynomial (degree 2) to the scaled allele frequencies
    #poly_coeffs = np.polyfit(t[ikeep], scaled_af, deg=2)

    # Extract the quadratic coefficient (the third element in poly_coeffs)
    #quadratic_coef = poly_coeffs[0]

    # Perform a linear regression on the quadratic term for p-value estimation
    #slope, intercept, r_value, p_value, std_err = linregress(t[ikeep] ** 2, scaled_af)

    # If the p-value is smaller than the threshold, return False
    if not np.isnan(p_value) and p_value < p_min:
        return False
    else:
        return True



#This function computes the consensus allele frequency trajectory by combining multiple
# replicates while accounting for coverage and potential bias correction.
def consensus_traj(traj, t, cov=None, Ne=None, N_sim=10000, bias=None):
    # Input validation
    if traj is None or len(traj) == 0:
        raise ValueError("No trajectory provided.")

    # Convert traj and cov to NumPy arrays
    traj = np.atleast_2d(traj)
    if cov is not None:
        cov = np.atleast_2d(cov)

    # Convert t to a NumPy array and validate
    t = np.asarray(t, dtype=float)
    if traj.shape[1] != len(t):
        raise ValueError("Number of columns in 'traj' must match the length of 't'.")

    if cov is not None and not np.all(np.isnan(cov)) and cov.shape[1] != len(t):
        raise ValueError("Number of columns in 'cov' must match the length of 't'.")

    if Ne is not None:
        Ne = float(Ne)
    N_sim = int(N_sim)

    # Output dictionary
    ctraj_out = {}

    # Single replicate case
    if traj.shape[0] == 1:
        # Sort trajectory by time and create output
        idx = np.argsort(t)
        ctraj_out["af"] = traj[0, idx]
        ctraj_out["t"] = t[idx]

        # Include coverage if provided
        if cov is not None and not np.all(np.isnan(cov)):
            ctraj_out["cov"] = cov[0, idx]

    # Multiple replicates case
    else:
        # Order allele frequencies by increasing time
        idx = np.argsort(t)
        traj = traj[:, idx]
        p0 = np.mean(traj[:, 0])

        # Mask replicates where allele is lost (cumulative max is 0)
        mask = np.maximum.accumulate(traj[:, ::-1], axis=1)[:, ::-1] == 0
        traj[mask] = np.nan

        # Compute mean allele frequency ignoring lost alleles
        ct = np.where(np.isnan(traj).all(axis=0), 0, np.nanmean(traj, axis=0))

        # Apply coverage masking if coverage is provided
        cc = None
        if cov is not None and not np.all(np.isnan(cov)):
            cov = cov[:, idx]
            cov[mask] = np.nan
            cc = np.where(np.isnan(cov).all(axis=0), 0, np.nansum(cov, axis=0))

        # Apply bias correction and ensure allele frequency is within [0, 1]
        if bias is None or np.all(np.isnan(bias)):
            bias = fixation_bias(traj=traj, t=t, Ne=Ne, N_sim=N_sim)
        ct -= bias
        ct = np.clip(ct, 0, 1)
        ct[0] = p0

        # Prepare output
        ctraj_out["af"] = ct
        ctraj_out["t"] = t[idx]
        if cc is not None:
            ctraj_out["cov"] = cc

    return ctraj_out





#This function aims to compute a bias correction for allele frequency trajectories, taking into account the fixation of
# alleles (when the allele frequency becomes 0 or 1).
def fixation_bias(traj, t, Ne, N_sim=10000):
    # Input validation
    if traj is None or len(traj) == 0:
        raise ValueError("No trajectory provided.")

    # Convert traj to a NumPy array and handle vector input
    traj = np.atleast_2d(traj)
    if traj.shape[0] == 1:
        return np.zeros(traj.shape[1])

    # If no simulations should be performed, return '0'
    if N_sim is None or N_sim < 1:
        return np.zeros(traj.shape[1])

    # Convert t and Ne to numeric values
    t = np.asarray(t, dtype=float)
    Ne = float(Ne)
    N_sim = int(N_sim)

    if traj.shape[1] != len(t):
        raise ValueError("Number of columns in 'traj' must match the length of 't'.")

    # Compute the mean allele frequency across replicates
    p_mean = np.nanmean(traj, axis=0)

    # If the initial allele frequency is 0, return a vector of zeros
    if p_mean[0] == 0:
        return np.zeros(len(p_mean))

    # Handle fixation events: Set allele frequency to NaN if it becomes 0
    traj = traj.copy()
    traj[:, ::-1][np.maximum.accumulate(traj[:, ::-1], axis=1) == 0] = np.nan
    p_mean_fix = np.where(np.isnan(traj).all(axis=0), 0, np.nanmean(traj, axis=0))

    t_diff = np.diff(t)
    pt_mean = p_mean[0]
    pt = np.full(N_sim, p_mean[0])
    bias = np.zeros(len(p_mean))

    # Loop through each time point to compute bias
    for i in range(len(p_mean) - 1):
        # If all pt values are zero, no bias correction is needed
        if len(pt) == 0:
            bias[i + 1] = bias[i]
            pt = np.full(N_sim, p_mean[i + 1])
        else:
            # Simulate allele frequencies after a time interval using genetic drift
            #pt = wf_traj(p0=pt, Ne=Ne, t=t_diff[i], s=0.0)
            pt=simulate(pt, Ne, t_diff[i], 1,s=0,h=0.5,n_round=8,n_sampling=1000,n_census=1000,coverage=None)

            # Remove zero values (alleles that are lost)
            pt = pt[pt != 0]

            # Compute bias correction
            if len(pt) == 0:
                bias[i + 1] = bias[i]
            else:
                bias[i + 1] = bias[i] + np.mean(pt) - pt_mean

            # Update allele frequencies based on observed mean allele frequency
            pt += p_mean_fix[i + 1] - np.mean(pt)
            pt = np.clip(pt, 0, 1)
            pt = pt[pt > 0]
            pt_mean = np.mean(pt)

    return bias



def estimate_sh(traj, t, Ne, haploid=False, h=np.nan, N_ctraj=0, cov=np.nan, approximate=True, method="LLS"):
    # Ensure traj is a NumPy array
    traj = np.atleast_2d(traj)
    if traj.shape[1] <= 1:
        raise ValueError("Allele frequencies at more than one time point have to be specified.")

    # Convert t and Ne to numeric
    t = np.asarray(t, dtype=float)
    Ne = float(Ne)

    if traj.shape[1] != len(t):
        raise ValueError("Number of columns in 'traj' has to be equal to the length of 't'.")
    if not np.isnan(h) and h != 0.5 and method == "LLS":
        raise ValueError("If LLS is used to estimate s, then h has to equal 0.5.")

    # Compute the consensus trajectory
    cBias = fixation_bias(traj=traj, t=t, Ne=Ne, N_sim=N_ctraj)
    cTraj = consensus_traj(traj=traj, t=t, cov=cov, bias=cBias)
    p0 = cTraj["af"][0]

    param = None
    used = ""


    if use_lls(cTraj,h=h,haploid=haploid):
        # Perform linear least-squares regression (LLS)
        param = lm_s(ctraj=cTraj, haploid=haploid, maxiter=0 if approximate else 10)
    else:
        # perform non-linear least-squares regression
        #nls.sh(ctraj=cTraj, Ne=Ne, h=h, haploid=haploid, s.start=0.1, h.start=0.5, approximate=approximate)
        param=nls_sh(ctraj=cTraj, Ne=Ne, h=h, haploid=haploid, s_start=0.1, h_start=0.5, approximate=approximate)
    used = "LLS"

    # Prepare result
    estsh_out = {
        "traj": traj,
        "t": t,
        "Ne": Ne,
        "haploid": haploid,
        "ctraj": cTraj,
        "N_ctraj": N_ctraj,
        "cov": cov if not np.all(np.isnan(cov)) else None,
        "method": method,
        "used": used,
        "s": param.get("s")
    }

    if "p0" in param:
        estsh_out["p0"] = param["p0"]

    if np.isnan(h):
        estsh_out["h_given"] = 0.5
    else:
        estsh_out["h_given"] = h

    return estsh_out

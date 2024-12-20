import logging
import math
import pickle
from os.path import exists
from random import sample

import h5py
import pandas
import soundfile as sf
from CQCC.CQT_toolbox_2013.cqt import cqt
from numpy import (arange, ceil, concatenate, errstate, exp, finfo, floor,
                   infty, linspace, log, meshgrid, sqrt, tile, vstack, zeros,
                   zeros_like)
from scipy.fft import dct
from scipy.interpolate import interpn
from scipy.signal import lfilter
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

# configs - init
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# feature extraction functions
def cqccDeltas(x, hlen=2):
    win = list(range(hlen, -hlen - 1, -1))
    norm = 2 * (arange(1, hlen + 1) ** 2).sum()
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx) / norm
    return D[:, hlen * 2 :]


def extract_cqcc(sig, fs, fmin, fmax, B=12, cf=19, d=16):
    # cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    kl = B * math.log(1 + 1 / d, 2)
    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))
    eps = 2.2204e-16
    scoeff = 1

    new_fs = 1 / (fmin * (2 ** (kl / B) - 1))
    ratio = 9.562 / new_fs

    Xcq = cqt(sig[:, None], B, fs, fmin, fmax, "rasterize", "full", "gamma", gamma)
    absCQT = abs(Xcq["c"])

    TimeVec = arange(1, absCQT.shape[1] + 1).reshape(1, -1)
    TimeVec = TimeVec * Xcq["xlen"] / absCQT.shape[1] / fs

    FreqVec = arange(0, absCQT.shape[0]).reshape(1, -1)
    FreqVec = fmin * (2 ** (FreqVec / B))

    LogP_absCQT = log(absCQT**2 + eps)

    n_samples = int(ceil(LogP_absCQT.shape[0] * ratio))
    Ures_FreqVec = linspace(FreqVec.min(), FreqVec.max(), n_samples)

    xi, yi = meshgrid(TimeVec[0, :], Ures_FreqVec)
    Ures_LogP_absCQT = interpn(
        points=(TimeVec[0, :], FreqVec[0, :]),
        values=LogP_absCQT.T,
        xi=(xi, yi),
        method="splinef2d",
    )

    CQcepstrum = dct(Ures_LogP_absCQT, type=2, axis=0, norm="ortho")
    CQcepstrum_temp = CQcepstrum[scoeff - 1 : cf + 1, :]
    deltas = cqccDeltas(CQcepstrum_temp.T).T

    CQcc = concatenate([CQcepstrum_temp, deltas, cqccDeltas(deltas.T).T], axis=0)

    return CQcc.T


def extract_features(file, features, cached=False):
    def get_feats():
        if features == "cqcc":
            return extract_cqcc(file)
        else:
            return None

    if cached:
        # cqcc is very slow, writing entire dataset to hdf5 file beforehand (offline cache)
        cache_file = features + ".h5"
        h5 = h5py.File(cache_file, "a")
        group = h5.get(file)
        if group is None:
            data = get_feats()
            h5.create_dataset(file, data=data, compression="gzip")
        else:
            data = group[()]
        h5.close()
        return data
    else:
        return get_feats()


def train_gmm(
    data_label,
    features,
    train_keys,
    train_folders,
    audio_ext,
    dict_file,
    ncomp,
    init_only=False,
):
    logging.info("Start GMM training.")

    partial_gmm_dict_file = "_".join((dict_file, data_label, "init", "partial.pkl"))
    if exists(partial_gmm_dict_file):
        gmm = GaussianMixture(covariance_type="diag")
        with open(partial_gmm_dict_file, "rb") as tf:
            gmm._set_parameters(pickle.load(tf))
    else:
        data = list()
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=" ", header=None)
            files = pd[pd[4] == data_label][1]
            # files_subset = sample(list(files), 1000)  # random init with 1000 files
            files_subset = (files.reset_index()[1]).loc[
                list(range(0, len(files), 10))
            ]  # only every 10th file init
            for file in files_subset:
                Tx = extract_features(
                    train_folders[k] + file + audio_ext, features=features, cached=True
                )
                data.append(Tx.T)

        X = vstack(data)
        gmm = GaussianMixture(
            n_components=ncomp,
            random_state=None,
            covariance_type="diag",
            max_iter=10,
            verbose=2,
            verbose_interval=1,
        ).fit(X)

        logging.info("GMM init done - llh: %.5f" % gmm.lower_bound_)

        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

    if init_only:
        return gmm

    # EM training
    prev_lower_bound = -infty
    for i in range(10):
        partial_gmm_dict_file = "_".join((dict_file, data_label, str(i), "partial.pkl"))
        if exists(partial_gmm_dict_file):
            with open(partial_gmm_dict_file, "rb") as tf:
                gmm._set_parameters(pickle.load(tf))
                continue

        nk_acc = zeros_like(gmm.weights_)
        mu_acc = zeros_like(gmm.means_)
        sigma_acc = zeros_like(gmm.covariances_)
        log_prob_norm_acc = 0
        n_samples = 0
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=" ", header=None)
            files = pd[pd[4] == data_label][1]

            for file in files.values:
                Tx = extract_features(
                    train_folders[k] + file + audio_ext, features=features, cached=True
                )
                n_samples += Tx.shape[1]

                # e step
                weighted_log_prob = gmm._estimate_weighted_log_prob(Tx.T)
                log_prob_norm = logsumexp(weighted_log_prob, axis=1)
                with errstate(under="ignore"):
                    # ignore underflow
                    log_resp = weighted_log_prob - log_prob_norm[:, None]
                log_prob_norm_acc += log_prob_norm.sum()

                # m step preparation
                resp = exp(log_resp)
                nk_acc += resp.sum(axis=0) + 10 * finfo(log(1).dtype).eps
                mu_acc += resp.T @ Tx.T
                sigma_acc += resp.T @ (Tx.T**2)

        # m step
        gmm.means_ = mu_acc / nk_acc[:, None]
        gmm.covariances_ = sigma_acc / nk_acc[:, None] - gmm.means_**2 + gmm.reg_covar
        gmm.weights_ = nk_acc / n_samples
        gmm.weights_ /= gmm.weights_.sum()
        if (gmm.covariances_ <= 0.0).any():
            raise ValueError("ill-defined empirical covariance")
        gmm.precisions_cholesky_ = 1.0 / sqrt(gmm.covariances_)

        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

        # infos
        lower_bound = log_prob_norm_acc / n_samples
        change = lower_bound - prev_lower_bound
        logging.info(
            "  Iteration %d\t llh %.5f\t ll change %.5f" % (i, lower_bound, change)
        )
        prev_lower_bound = lower_bound

        if abs(change) < gmm.tol:
            logging.info("  Coverged; too small change")
            gmm.converged_ = True
            break

    return gmm


def scoring(
    scores_file,
    dict_file,
    features,
    eval_ndx,
    eval_folder,
    audio_ext,
    features_cached=True,
    flag_debug=False,
):
    logging.info("Scoring eval data")

    gmm_bona = GaussianMixture(covariance_type="diag")
    gmm_spoof = GaussianMixture(covariance_type="diag")
    with open(dict_file, "rb") as tf:
        gmm_dict = pickle.load(tf)
        gmm_bona._set_parameters(gmm_dict["bona"])
        gmm_spoof._set_parameters(gmm_dict["spoof"])

    pd = pandas.read_csv(eval_ndx, sep=" ", header=None)
    if flag_debug:
        pd = pd[:1000]

    files = pd[1].values
    scr = zeros_like(files, dtype=log(1).dtype)
    for i, file in enumerate(files):
        if (i + 1) % 1000 == 0:
            logging.info("\t...%d/%d..." % (i + 1, len(files)))

        try:
            Tx = extract_features(
                eval_folder + file + audio_ext,
                features=features,
                cached=features_cached,
            )
            scr[i] = gmm_bona.score(Tx.T) - gmm_spoof.score(Tx.T)
        except Exception as e:
            logging.warning(e)
            scr[i] = log(1)

    pd_out = pandas.DataFrame({"files": files, "scores": scr})
    pd_out.to_csv(scores_file, sep=" ", header=False, index=False)

    logging.info("\t... scoring completed.\n")


def scoring_partials(
    scores_file,
    dict_dict_files,
    features,
    eval_ndx,
    eval_folder,
    audio_ext,
    features_cached=True,
    flag_debug=False,
):
    logging.info("Scoring eval data")

    gmms = dict()
    for i, dict_files in dict_dict_files.items():
        gmms[i] = dict()
        gmms[i]["bona"] = GaussianMixture(covariance_type="diag")
        gmms[i]["spoof"] = GaussianMixture(covariance_type="diag")

        with open(dict_files["bona"], "rb") as tf:
            gmm_dict = pickle.load(tf)
            gmms[i]["bona"]._set_parameters(gmm_dict)

        with open(dict_files["spoof"], "rb") as tf:
            gmm_dict = pickle.load(tf)
            gmms[i]["spoof"]._set_parameters(gmm_dict)

    pd = pandas.read_csv(eval_ndx, sep=" ", header=None)
    if flag_debug:
        pd = pd[:1000]

    files = pd[1].values
    scr = zeros((files.shape[0], len(gmms.keys())), dtype=log(1).dtype)
    for i, file in enumerate(files):
        if (i + 1) % 1000 == 0:
            logging.info("\t...%d/%d..." % (i + 1, len(files)))

        try:
            Tx = extract_features(
                eval_folder + file + audio_ext,
                features=features,
                cached=features_cached,
            )
            for j, jkey in enumerate(gmms.keys()):
                scr[i, j] = gmms[j]["bona"].score(Tx.T) - gmms[j]["spoof"].score(Tx.T)
        except Exception as e:
            logging.warning(e)
            for j, _ in enumerate(gmms.keys()):
                scr[i, j] = log(1)

    pd_out = pandas.DataFrame(scr).set_index(files)
    pd_out.to_csv(scores_file, sep=" ", header=False, index=True)

    logging.info("\t... scoring completed.\n")

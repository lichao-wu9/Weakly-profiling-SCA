import numpy as np
import random
import scipy.stats as ss
import Util.SCA_dataset as datasets

def rk_key_eff(rank_array, key):
    return np.argsort(np.argsort(rank_array, axis=1), axis=1)[:, key]

def plt_dist(leakage_model):
    container = np.zeros((256, 256, 256), dtype=np.int16)
    for k in range(256):
        if leakage_model == "HW":
            label = datasets.calculate_HW(datasets.AES_Sbox[np.bitwise_xor(range(256), k)])
        else:
            label = datasets.AES_Sbox[np.bitwise_xor(range(256), k)]
        for plt in range(256):
            container[k, plt] = ss.rankdata(np.power(label[plt] - label, 2), method='dense') - 1
    return container

def perform_attacks(predictions, plaintext, byte, nb_attacks, num_of_attack_traces, correct_key, shuffle=True, leakage_model="HW"):
    plt_dist_all_key = plt_dist(leakage_model)
    all_rk_evol = np.zeros((nb_attacks, num_of_attack_traces))
    accu_pred = np.zeros((256, 256))
    corr_for_each_plt = np.zeros((256, 256))
    corr_container = np.zeros((num_of_attack_traces, 256))
    plt_attack = plaintext[:, byte]

    for idx in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            pred = sp[:num_of_attack_traces]
            plt = splt[:num_of_attack_traces]
        else:
            pred = predictions[:num_of_attack_traces]
            plt = plt_attack[:num_of_attack_traces]

        for i in range(num_of_attack_traces):
            accu_pred[plt[i]] += pred[i]
            accu_pred_rank = ss.rankdata(accu_pred[plt[i]], method='dense') - 1
            corr_for_each_plt[plt[i]] = np.corrcoef(accu_pred_rank, plt_dist_all_key[:, plt[i]])[0][1:]
            corr_container[i] = np.sum(corr_for_each_plt, axis=0)

        # Compute and plot the rank of the correct delta
        all_rk_evol[idx] = rk_key_eff(corr_container, correct_key)
    return np.mean(all_rk_evol, axis=0)
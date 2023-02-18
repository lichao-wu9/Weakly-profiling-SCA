from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras import backend as K

# for visualizing
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler

import Util.SCA_dataset as datasets
import Util.DL_model as DL_model
from Util.one_cycle_lr import OneCycleLR
import Util.Attack as Attack

if __name__ == "__main__":
    data_root = ''
    model_root = ''
    result_root = ''

    # the dataset to be tested
    datasetss = ['ASCAD'] 
    # leakage models
    leakage_models = ["HW", "ID"]
    # data augmentaiton level
    aug_level = float(10)
    # Noise level
    noise_level = 0
    # Training epochs
    epochs = 250
    # True if we want to train a new model; False if we want to load an existing model
    train_model = True
    # Naming index
    index = 1

    for dataset in datasetss:
        if dataset == 'ASCAD':
            target_byte = 2
            profiling_num = 30000
            all_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]
            target_key_0 = all_key[target_byte]
            X_profiling_o, _, plt_profiling_o = datasets.load_ascad(data_root+dataset+'/', target_byte)
        elif dataset == 'ASCAD_rand':
            target_byte = 2
            profiling_num = 30000
            all_key = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]
            target_key_0 = all_key[target_byte]
            X_profiling_o, _, plt_profiling_o = datasets.load_ascad_rand(data_root+dataset+'/', target_byte)
        elif dataset == 'CHES_CTF':
            target_byte = 0
            profiling_num = 40000
            all_key = [23, 92, 242, 153, 122, 133, 131, 65, 60, 119, 223, 172, 126, 108, 89, 216]
            target_key_0 = all_key[target_byte]
            X_profiling_o, _, plt_profiling_o = datasets.load_chesctf(data_root+dataset+'/', target_byte)
        elif dataset == 'AESHD':
            target_byte = 7 # Sbox_inv(k7^c7)^c3
            profiling_num = 30000
            all_key = [208, 20, 249, 168, 201, 238,  37, 137, 225,  63,  12, 200, 182, 99, 12, 166]
            target_key_0 = all_key[target_byte]
            X_profiling_o, _, plt_profiling_o = datasets.load_aeshd(data_root+dataset+'/', target_byte)
        elif dataset == 'AESRD':
            target_byte = 0
            profiling_num = 30000
            all_key = [43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60]
            target_key_0 = all_key[target_byte]
            X_profiling_o, _, plt_profiling_o = datasets.load_aesrd(data_root+dataset+'/', target_byte)

        # add noise if needed
        X_profiling_o = datasets.addDesync(X_profiling_o, noise_level)
        X_profiling = X_profiling_o[:int(profiling_num)]
        plt_profiling = plt_profiling_o[:int(profiling_num)]

        # Normalize the data
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)

        # Test info
        test_info = '{}_{}_{}_{}_epoch{}_byte{}_{}'.format(dataset, profiling_num, aug_level, noise_level, epochs, target_byte, index)
        print('====={}====='.format(test_info))
        
        # if model is CNN, we have to make the data dim equals to 3 
        X_profiling = np.expand_dims(X_profiling, axis=-1)

        # load and train model if needed
        if train_model:
            model, batch_size = DL_model.CNN(dataset, X_profiling.shape[1], aug_level)
            callback = [OneCycleLR(len(X_profiling), batch_size, 5e-3, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=True)]
            model.fit(
                x=X_profiling, 
                y=to_categorical(plt_profiling[:, target_byte], num_classes=256),
                batch_size=batch_size, 
                verbose=2, 
                epochs=epochs,
                callbacks=callback)           

            model.save(model_root+"model_sbox{}_{}.h5".format(target_byte, test_info))
        else:
            model = load_model(model_root+"model_sbox{}_{}.h5".format(target_byte, test_info))
        
        # ================================================================
        num_of_attack_traces = len(X_profiling)
        
        # disable the augmentation layer and load the model weight
        model_test, _ = DL_model.CNN(dataset, X_profiling.shape[1], 0)
        model_test.set_weights(model.get_weights()) 
        
        # make predictions
        pred = model_test.predict(X_profiling)
        save_container = np.zeros((len(leakage_models), num_of_attack_traces))
        
        # perform attacks for each leakage model
        for i, leakage_model in enumerate(leakage_models):
            print("======{}======".format(leakage_model))
            rank_evol = Attack.perform_attacks(pred, plt_profiling, target_byte, 1, num_of_attack_traces, target_key_0, leakage_model=leakage_model, shuffle=False)
            print("Final rank: ", rank_evol[-1])
            print('GE smaller than 1:', np.argmax(rank_evol < 1))
            print('GE smaller than 5:', np.argmax(rank_evol < 5))
            plt.plot(rank_evol, label = "Sbox{}_{}".format(target_byte, leakage_model))       
            save_container[i] = rank_evol
            
        plt.legend()        
        plt.savefig(result_root + "Key_Guess_{}.png".format(test_info))
        # plt.show()
        plt.clf() 
        np.save(result_root + "Key_Guess_{}.npy".format(test_info), save_container)
        K.clear_session()

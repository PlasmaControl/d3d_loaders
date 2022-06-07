import numpy as np
import pickle
import glob
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import pandas as pd
import matplotlib.patches as mpatches
import scipy.sparse
import scipy.signal
from scipy.sparse.linalg.eigen.arpack import eigs as Eigens
# import seaborn as sns
import random
import h5py
import scipy.interpolate as interp
#import cv2

# from plotly import figure_factory as ff

##############################################################################

def init_weights(input_dim, res_size, K_in, K_rec, insca, spra, bisca):
    # ---------- Initializing W_in ---------
    winexist, fname = find_saved_weights('Win', input_dim, res_size, K_in, K_rec)
    if winexist:
        with open(fname, 'rb') as f:
            W_in = pickle.load(f)
            W_in *= insca
    else:
        if K_in == -1 or input_dim < K_in:
            W_in = insca * (np.random.rand(res_size, input_dim) * 2 - 1)
        else:
            Ico = 0
            nrentries = np.int32(res_size * K_in)
            ij = np.zeros((2, nrentries))
            datavec = insca * (np.random.rand(nrentries) * 2 - 1)
            for en in range(res_size):
                Per = np.random.permutation(input_dim)[:K_in]
                ij[0][Ico:Ico + K_in] = en
                ij[1][Ico:Ico + K_in] = Per
                Ico += K_in
            W_in = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, input_dim), dtype='float64')
            if K_in > input_dim / 2:
                W_in = W_in.todense()

    # ---------- Initializing W_res ---------
    wrecexist, fname = find_saved_weights('Wres', input_dim, res_size, K_in, K_rec)
    if wrecexist:
        with open(fname, 'rb') as f:
            W_res = pickle.load(f)
            W_res *= spra
    else:
        converged = False
        attempts = 50
        while not converged and attempts > 0:
            if K_rec == -1:
                W_res = np.random.randn(res_size, res_size)
            else:
                Ico = 0
                nrentries = np.int32(res_size * K_rec)
                ij = np.zeros((2, nrentries))
                datavec = np.random.randn(nrentries)
                for en in range(res_size):
                    Per = np.random.permutation(res_size)[:K_rec]
                    ij[0][Ico:Ico + K_rec] = en
                    ij[1][Ico:Ico + K_rec] = Per
                    Ico += K_rec
                W_res = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, res_size), dtype='float64')
                if K_rec > res_size / 2:
                    W_res = W_res.todense()
            try:
                we = Eigens(W_res, return_eigenvectors=False, k=6)
                converged = True
            except:
                print("WARNING: No convergence! Redo %i times ... " % (attempts - 1))
                attempts -= 1
                pass

        W_res *= (spra / np.amax(np.absolute(we)))
    # ---------- Initializing W_bi ---------
    wbiexist, fname = find_saved_weights('Wbi', input_dim, res_size, K_in, K_rec)
    if wbiexist:
        with open(fname, 'rb') as f:
            W_bi = pickle.load(f)
            W_bi *= bisca
    else:
        W_bi = bisca * (np.random.rand(res_size) * 2 - 1)
    print("found (W_in, W_rec, W_bi) > (%s, %s, %s)" % (winexist, wrecexist, wbiexist))
    return W_in, W_res, W_bi


##############################################################################
def find_saved_weights(wname, input_dim, res_size, K_in, K_rec):
    s_r = []
    saved_reservoirs = glob.glob('/scratch/gpfs/aj17/saved_files/saved_' + wname + '*')
    for curr_file in (saved_reservoirs):
        s_r.append(curr_file[curr_file.rfind('_') + 1:curr_file.rfind('.')])
    if wname == 'Win':
        required_file_id = 'I' + str(input_dim) + 'R' + str(res_size) + 'Kin' + str(K_in)
    elif wname == 'Wres':
        required_file_id = 'R' + str(res_size) + 'Krec' + str(K_rec)
    elif wname == 'Wbi':
        required_file_id = 'R' + str(res_size)
    return (required_file_id in s_r), '/scratch/gpfs/aj17/saved_files/saved_' + wname + '_' + required_file_id + '.pkl'


##############################################################################

def res_exe(W_in, W_res, W_bi, leak, U):
    T = U.shape[0]  # size of the input vector
    nres = W_res.shape[0]  # Getting the size of the network (= 100)
    R = np.zeros((T + 1, nres),
                 dtype='float64')  # Initializing the RCN output matrix (one extra frame for the warming up)
    for t in range(T):  # for each frame
        if scipy.sparse.issparse(W_in):
            a = W_in * U[t, :]
        else:
            a = np.dot(W_in, U[t, :])

        if scipy.sparse.issparse(W_res):
            b = W_res * R[t, :]
        else:
            b = np.dot(W_res, R[t, :])
        R[t + 1, :] = np.tanh(a + b + W_bi)
        R[t + 1, :] = (1 - leak) * R[t, :] + leak * R[t + 1, :]
    R = np.concatenate((np.ones((R.shape[0], 1)), R), 1)
    return R[1:, :]  # returns the reservoir output and the desired output


##############################################################################

def res_train(xTx, xTy, xlen, regu):
    t1 = time.time()
    lmda = regu ** 2 * xlen
    inv_xTx = np.linalg.inv(xTx + lmda * np.eye(xTx.shape[0]))
    beta = np.dot(inv_xTx, xTy)  # beta is the output weight matrix
    # print ('RCN trained\tin '+str(round(time.time()-t1,2))+' sec.!')
    return beta


##############################################################################

# def calderiv(c, arg=[2, 1, 2]):
    # Vlen, Alen, w = arg
    # nf = c.shape[0]
    # nc = c.shape[1]
    # dv = denom_delta(Vlen)
    # da = denom_delta(Alen)
    # if w == 2:
        # vf = np.array(range(Vlen, -(Vlen + 1), -1)) / dv
        # af = np.array(range(Alen, -(Alen + 1), -1)) / da
        # cx = np.vstack((np.tile(c[0], [Vlen + Alen, 1]), c, np.tile(c[-1], [Vlen + Alen, 1])))
        # vx = np.reshape(scipy.signal.lfilter(vf, 1, cx.flatten(1)), (nf + 2 * (Vlen + Alen), nc), order='F')
        # vx = np.delete(vx, range(2 * Vlen), 0)
        # ax = np.reshape(scipy.signal.lfilter(af, 1, vx.flatten(1)), (nf + 2 * Alen, nc), order='F')
        # ax = np.delete(ax, range(2 * Alen), 0)
        # vx = vx[Alen:nf + Alen, :]
        # # if w.find('d')!=-1:
        # #    c=np.hstack((c,vx,ax))
        # # else:
        # #    c=np.hstack((c,ax))
        # c = np.hstack((c, vx, ax))
    # elif w == 1:
        # vf = np.array(range(Vlen, -(Vlen + 1), -1)) / dv
        # cx = np.vstack((np.tile(c[0], [Vlen, 1]), c, np.tile(c[-1], [Vlen, 1])))
        # vx = np.reshape(scipy.signal.lfilter(vf, 1, cx.flatten(1)), (nf + 2 * Vlen, nc), order='F')
        # vx = np.delete(vx, range(2 * Vlen), 0)
        # c = np.hstack((c, vx))
    # return c


##############################################################################

# def denom_delta(N):
    # d = 0
    # for i in range(N + 1):
        # d += pow(i, 2)
    # return float(2 * d)


##############################################################################

def file_append(nfile, txline):
    fid = open(nfile, "a")
    fid.write(txline)
    fid.close()
    return 1


##############################################################################

def frame_stacking(I, frwin=[0, 0, '0']):
    if frwin[0] + frwin[1] == 0:
        return I
    else:
        if frwin[2] == 'm':
            B = np.hamming(2 * frwin[0] + 1)[:frwin[0]]
            A = np.hamming(2 * frwin[1] + 1)[frwin[1] + 1:]
        elif frwin[2] == 'n':
            B = np.hanning(2 * frwin[0] + 1)[:frwin[0]]
            A = np.hanning(2 * frwin[1] + 1)[frwin[1] + 1:]
        else:
            B = np.ones(frwin[0])
            A = np.ones(frwin[1])

        n_fr, n_fe = I.shape
        tmp = np.ones((n_fr + frwin[0] + frwin[1], n_fe))
        tmp[:frwin[0], :] = I[0, :]
        if frwin[1] != 0:
            tmp[-frwin[1]:, :] = I[-1, :]
        tmp[frwin[0]:frwin[0] + n_fr, :] = I
        t_f_w = []
        for i in range(frwin[0], frwin[0] + n_fr):
            tt = (tmp[i - frwin[0]:i, :].T * B).T
            bef = np.reshape(tt, (1, -1))[0]
            tt = (tmp[i + 1:i + frwin[1] + 1, :].T * A).T
            aft = np.reshape(tt, (1, -1))[0]
            t_f_w.append(np.concatenate((bef, tmp[i, :], aft), 0))
        t_f_w = np.array(t_f_w)
        return t_f_w


##############################################################################

def run_sample(inputs, n_fr_pred, input_ind, output_ind, frm_stck, cal_drv, W_in1, W_res1, W_bi1, beta1, W_in2, W_res2,
               W_bi2, beta2, leak):
    u = calderiv(frame_stacking(inputs[:-n_fr_pred, input_ind], frm_stck), cal_drv)
    d = inputs[n_fr_pred:, output_ind] - inputs[:-n_fr_pred, output_ind]
    x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)
    o1 = np.dot(x1, beta1)
    x2 = res_exe(W_in2, W_res2, W_bi2, leak[1], o1)
    o2 = np.dot(x2, beta2)

    for i, j in zip(range(5), ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']):
        a = plt.subplot(5, 1, i + 1)
        a = plt.plot(np.mean(d[:, i * 65:(i + 1) * 65], axis=1), 'r')
        a = plt.plot(np.mean(o1[:, i * 65:(i + 1) * 65], axis=1), 'c--', linewidth=1)
        a = plt.plot(np.mean(o2[:, i * 65:(i + 1) * 65], axis=1), 'b')
        plt.ylabel(j)
    plt.legend(['Target', 'RCN_L1', 'RCN_L2'])
    plt.show()
    return ([o1, o2, d])


##############################################################################

def evaluate(fld_names, main_data_path, n_fr_pred, input_ind, output_ind, frm_stck, cal_drv, nread, W_in1, W_res1,
             W_bi1, beta1, leak, beta_reg):
    profiles = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    a = [os.remove(i) for i in glob.glob(main_data_path + 'eval/' + fld_names + '_*.pkl')]

    mse = np.zeros((325, 6))

    shotnum_list = [k.split('\\')[-1].split('.')[0] for k in sorted(glob.glob(main_data_path + fld_names + '/*.pkl'))]
    # if fld_names == 'test':
    #     lstm_shot_names = [k.split('\\')[-1].split('.')[0] for k in glob.glob('../BOF_data/LSTM_prediction/test/*.pkl')]
    #     shotnum_list = list(set(lstm_shot_names) & set(shotnum_list))
    #     # a = [shotnum_list.remove(k) for k in ['156520', '162720', '164280', '170590', '172380', '175970']]
    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]
    too_short = 0
    mean_mse = np.zeros((len(shotnum_list), 4))
    # for idx, shotnum in tqdm(enumerate(shotnum_list), desc='Evaluate' + fld_names, ascii=True,
    #                          dynamic_ncols=True):
    for idx, shotnum in enumerate(shotnum_list):
        raw_data = pickle.load(open(main_data_path + fld_names + '/' + shotnum + '.pkl', 'rb'))
        mean_mse[idx, 0] = shotnum
        if (len(raw_data['shotnum']) > n_fr_pred):
            inputs = 0
            for keys, values in raw_data.items():
                if keys != 'shotnum':
                    if type(inputs) == int:
                        inputs = np.reshape(np.array(values), (np.array(values).shape[0], -1))
                    else:
                        inputs = np.hstack((inputs, np.reshape(np.array(values), (inputs.shape[0], -1))))

            inputs_tmp = inputs.copy()
            inputs_tmp[:-n_fr_pred, -4:] = inputs_tmp[n_fr_pred:, -4:]  # include actuators of the future
            u = calderiv(frame_stacking(inputs_tmp[:-n_fr_pred, input_ind], frm_stck), cal_drv)
            x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)
            o1 = np.dot(x1, beta1)
            rcn_targ = inputs[n_fr_pred:, output_ind]
            rcn_pred_l1 = inputs[:-n_fr_pred, output_ind] + o1  # [:-n_fr_pred, :]

            u_reg = np.concatenate((np.ones((u.shape[0], 1)), u), 1)
            o_reg = np.dot(u_reg, beta_reg)
            lin_reg_pred = inputs[:-n_fr_pred, output_ind] + o_reg

            mse[:, 0] += np.sum(np.power(inputs[:-n_fr_pred, output_ind] - rcn_targ, 2), axis=0)
            mse[:, 1] += np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)
            mse[:, 2] += np.sum(np.power(lin_reg_pred - rcn_targ, 2), axis=0)
            mse[:, 3] += np.sum(np.power(rcn_targ, 2), axis=0)
            mse[:, 4] += np.sum(rcn_targ, axis=0)
            mse[:, 5] += rcn_targ.shape[0]

            mean_mse[idx, 1] = np.mean(
                np.sqrt(np.sum(np.power(inputs[:-n_fr_pred, output_ind] - rcn_targ, 2), axis=0)) / sd)
            mean_mse[idx, 2] = np.mean(np.sqrt(np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)) / sd)
            mean_mse[idx, 3] = np.mean(np.sqrt(np.sum(np.power(lin_reg_pred - rcn_targ, 2), axis=0)) / sd)

            with open(main_data_path + 'eval/' + fld_names + '_output_' + str(np.int32(raw_data['shotnum'][0])) + '.pkl',
                      'wb') as f:
                pickle.dump(mse, f)
            np.savetxt(main_data_path + 'eval/noadapt_' + fld_names + '_output_' + str(
                np.int32(raw_data['shotnum'][0])) + '.txt',
                       np.hstack((rcn_targ, inputs[:-n_fr_pred, output_ind], rcn_pred_l1, lin_reg_pred)))
        else:
            too_short += 1

    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]

    rmse_baseline = np.sqrt(mse[:, 0] / mse[:, 5])
    nrmse_baseline = rmse_baseline / sd

    rmse_rcn_l1 = np.sqrt(mse[:, 1] / mse[:, 5])
    nrmse_rcn_l1 = rmse_rcn_l1 / sd

    rmse_rcn_l2 = np.sqrt(mse[:, 2] / mse[:, 5])
    nrmse_rcn_l2 = rmse_rcn_l2 / sd

    print('%i out of %i (%.1f%%) were shorter than %i frames!' % (
    too_short, len(shotnum_list), too_short * 100.0 / len(shotnum_list), n_fr_pred))
    Rep = 'AvgNRMSE: Base,L1(%i),LinReg -> %.3e, %.3e, %.3e\n' % (
    W_res1.shape[0], np.mean(nrmse_baseline), np.mean(nrmse_rcn_l1), np.mean(nrmse_rcn_l2))
    print(Rep)
    return mse, mean_mse, Rep


##############################################################################

def PCA_evaluation(fld_names, main_data_path, n_fr_pred, input_ind, output_ind, frm_stck, cal_drv, nread, W_in1, W_res1,
                   W_bi1, beta1, W_in2, W_res2, W_bi2, beta2, leak):
    profiles = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    MSE = np.zeros((nread + 2, 2))

    file_names = glob.glob(main_data_path + fld_names + '/*.pkl')
    file_names.sort()
    too_short = 0
    Target = np.zeros((1, 325))
    Output = np.zeros((1, 325))
    for fname in file_names:  # mdp.utils.progress_bar(tr_file_names):

        raw_data = pickle.load(open(fname, 'rb'))
        if (len(raw_data['shotnum']) > n_fr_pred):
            inputs = 0
            for keys, values in raw_data.items():
                if keys != 'shotnum':
                    if type(inputs) == int:
                        inputs = np.reshape(np.array(values), (np.array(values).shape[0], -1))
                    else:
                        inputs = np.hstack((inputs, np.reshape(np.array(values), (inputs.shape[0], -1))))

            u = calderiv(frame_stacking(inputs[:-n_fr_pred, input_ind], frm_stck), cal_drv)
            d = inputs[n_fr_pred:, output_ind] - inputs[:-n_fr_pred, output_ind]
            x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)
            o1 = np.dot(x1, beta1)
            x2 = res_exe(W_in2, W_res2, W_bi2, leak[1], o1)
            o2 = np.dot(x2, beta2)
            Target = np.vstack((Target, d))
            Output = np.vstack((Output, o1))
        else:
            # print(str(raw_data['shotnum'][0])+' too short!' )
            too_short += 1
    print('%i out of %i (%.1f%%) were shorter than %i frames!' % (
        too_short, len(file_names), too_short * 100.0 / len(file_names), n_fr_pred))
    return (Target, Output)


def rcn_lstm_plot(fld, fname, n_fr_pred, channel_id):
    U_rcn, tmp = pickle.load(open('../BOF_data/eval/' + fld + '_output_' + fname + '.pkl', 'rb'))
    U_lstm = pickle.load(open('../BOF_data/LSTM_prediction/' + fld + '/' + fname + '.pkl', 'rb'))
    profile_list = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    xtick = np.round(U_rcn['dens_target'].shape[0] / 10)
    for idx, prf in enumerate(profile_list):
        rcn_targ = U_rcn[prf + '_target'][:, ::2]
        rcn_pred_l1 = U_rcn[prf + '_pred_L1'][:, ::2]
        rcn_pred_l2 = U_rcn[prf + '_pred_L2'][:, ::2]

        lstm_inp = np.array(U_lstm[prf])
        lstm_pred = np.array(U_lstm[prf + '_prediction'])

        curr_lstm_targ = lstm_inp[6 + n_fr_pred:, :]
        curr_lstm_pred = np.sum((lstm_inp[6:-n_fr_pred, :], lstm_pred[6:-n_fr_pred, :]), axis=0)
        np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)
        # plt.subplot(5,1,idx+1)
        # plt.plot(rcn_targ[:, channel_id])
        # plt.plot(curr_lstm_targ[:,channel_id])
        # plt.ylim([-1, 1])
        # plt.xticks(ticks=np.arange(xtick)*10,labels=np.arange(xtick)*0.5)
        plt.subplot(5, 1, idx + 1)
        plt.plot(rcn_targ[:, channel_id])
        plt.plot(rcn_pred_l1[:, channel_id])
        plt.plot(curr_lstm_pred[:, channel_id])

        plt.ylabel(prf)

        # plt.ylim([-1,1])
        plt.xticks(ticks=np.arange(xtick) * 10, labels=np.arange(xtick) * 0.5)
    plt.legend(['Target', 'RCN', 'LSTM'], loc='right')
    plt.xlabel('time (s)')
    plt.show()
    return U_rcn, U_lstm


def lstm_eval(n_fr_pred):
    profile_list = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    n_chann = 33
    mse = np.zeros((n_chann * len(profile_list), 5))
    rcn_shot_names = [k.split('_')[-1].split('.')[0] for k in glob.glob('../BOF_data/eval/test_*.pkl')]
    lstm_shot_names = [k.split('\\')[-1].split('.')[0] for k in glob.glob('../BOF_data/LSTM_prediction/test/*.pkl')]
    final_shotnums = list(set(lstm_shot_names) & set(rcn_shot_names))
    for shotnum in final_shotnums:

        U_lstm = pickle.load(open('../BOF_data/LSTM_prediction/test/' + shotnum + '.pkl', 'rb'))

        for idx, prf in enumerate(profile_list):
            lstm_inp = np.array(U_lstm[prf])
            lstm_pred = np.array(U_lstm[prf + '_prediction'])

            curr_lstm_targ = lstm_inp[6 + n_fr_pred:, :]
            curr_lstm_pred = np.sum((lstm_inp[6:-n_fr_pred, :], lstm_pred[6:-n_fr_pred, :]), axis=0)

            mse[idx * n_chann:(idx + 1) * n_chann, 0] += np.sum(np.power(lstm_inp[6:-n_fr_pred, :] - curr_lstm_targ, 2),
                                                                axis=0)
            mse[idx * n_chann:(idx + 1) * n_chann, 1] += np.sum(np.power(curr_lstm_pred - curr_lstm_targ, 2), axis=0)
            mse[idx * n_chann:(idx + 1) * n_chann, 2] += np.sum(np.power(curr_lstm_targ, 2), axis=0)
            mse[idx * n_chann:(idx + 1) * n_chann, 3] += np.sum(curr_lstm_targ, axis=0)
            mse[idx * n_chann:(idx + 1) * n_chann, 4] += lstm_pred[6:-n_fr_pred, 0].shape[0]
    sd = np.loadtxt('../sd2_lstm.txt')[:325]

    rmse_baseline = np.sqrt(mse[:, 0] / mse[:, 4])
    nrmse_baseline = rmse_baseline / sd

    rmse_lstm = np.sqrt(mse[:, 1] / mse[:, 4])
    nrmse_lstm = rmse_lstm / sd

    Rep = 'Mean-NRMSE: Baseline, LSTM (%i Channels!) -> %.3e, %.3e\n' % (
    n_chann, np.mean(nrmse_baseline), np.mean(nrmse_lstm))
    print(Rep)
    return mse


##############################################################################
def adaptation(fld_names, main_data_path, n_fr_pred, input_ind, output_ind, frm_stck, cal_drv, nread, W_in1, W_res1,
               W_bi1, beta1, leak, beta_reg, xTx, xTy, xlen, regu):
    xTx_ad = np.zeros((W_res1.shape[0] + 1, W_res1.shape[0] + 1), dtype='float64')
    xTy_ad = np.zeros((W_res1.shape[0] + 1, nread), dtype='float64')
    xlen_ad = 0
    beta1_new = beta1.copy()
    adapt_ind = 0
    final_mse = []
    adapt_time = 10
    profiles = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    a = [os.remove(i) for i in glob.glob(main_data_path + 'eval/' + fld_names + '_*.pkl')]

    mse = np.zeros((325, 6))

    shotnum_list = [k.split('\\')[-1].split('.')[0] for k in sorted(glob.glob(main_data_path + fld_names + '/*.pkl'))]
    # if fld_names == 'test':
    #     lstm_shot_names = [k.split('\\')[-1].split('.')[0] for k in glob.glob('../BOF_data/LSTM_prediction/test/*.pkl')]
    #     shotnum_list = list(set(lstm_shot_names) & set(shotnum_list))
    #     # a = [shotnum_list.remove(k) for k in ['156520', '162720', '164280', '170590', '172380', '175970']]
    too_short = 0
    mean_mse = np.zeros((len(shotnum_list), 2))
    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]
    tm = np.zeros(2)
    # for idx in tqdm(range(len(shotnum_list) - 1), desc='Adapt', ascii=True,
    #                 dynamic_ncols=True):
    for idx in range(len(shotnum_list) - 1):
        tr_shot = shotnum_list[idx]
        vl_shot = shotnum_list[idx + 1]
        raw_data = pickle.load(open(main_data_path + fld_names + '/' + tr_shot + '.pkl', 'rb'))
        if (len(raw_data['shotnum']) > n_fr_pred):

            inputs = 0
            for keys, values in raw_data.items():
                if keys != 'shotnum':
                    if type(inputs) == int:
                        inputs = np.reshape(np.array(values), (np.array(values).shape[0], -1))
                    else:
                        inputs = np.hstack((inputs, np.reshape(np.array(values), (inputs.shape[0], -1))))

            inputs_tmp = inputs.copy()
            inputs_tmp[:-n_fr_pred, -4:] = inputs_tmp[n_fr_pred:, -4:]  # include actuators of the future
            u = calderiv(frame_stacking(inputs_tmp[:-n_fr_pred, input_ind], frm_stck), cal_drv)

            d = inputs[n_fr_pred:, output_ind] - inputs[:-n_fr_pred, output_ind]
            t1 = time.time()
            x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)
            xlen_ad += u.shape[0]
            xTx_ad += np.dot(x1.T, x1)
            xTy_ad += np.dot(x1.T, d)
            beta1_new = res_train(xTx + xTx_ad, xTy + xTy_ad, xlen + xlen_ad, regu[0])
            # if adapt_ind > 10:
            #     beta1_new = res_train(xTx+xTx_ad, xTy+xTy_ad, xlen+xlen_ad, regu[0])
            #     adapt_ind =0
            # else:
            #     adapt_ind+=1
            tm[0] += (time.time() - t1)
            tm[1] += 1
            # --------
            raw_data = pickle.load(open(main_data_path + fld_names + '/' + vl_shot + '.pkl', 'rb'))
            if (len(raw_data['shotnum']) > n_fr_pred):
                inputs = 0
                for keys, values in raw_data.items():
                    if keys != 'shotnum':
                        if type(inputs) == int:
                            inputs = np.reshape(np.array(values), (np.array(values).shape[0], -1))
                        else:
                            inputs = np.hstack((inputs, np.reshape(np.array(values), (inputs.shape[0], -1))))

                inputs_tmp = inputs.copy()
                inputs_tmp[:-n_fr_pred, -4:] = inputs_tmp[n_fr_pred:, -4:]  # include actuators of the future
                u = calderiv(frame_stacking(inputs_tmp[:-n_fr_pred, input_ind], frm_stck), cal_drv)

                x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)
                o1 = np.dot(x1, beta1)
                o1_new = np.dot(x1, beta1_new)
                rcn_targ = inputs[n_fr_pred:, output_ind]
                rcn_pred_l1 = inputs[:-n_fr_pred, output_ind] + o1  # [:-n_fr_pred, :]
                rcn_pred_l1_new = inputs[:-n_fr_pred, output_ind] + o1_new  # [:-n_fr_pred, :]

                u_reg = np.concatenate((np.ones((u.shape[0], 1)), u), 1)
                o_reg = np.dot(u_reg, beta_reg)
                lin_reg_pred = inputs[:-n_fr_pred, output_ind] + o_reg

                mse[:, 0] += np.sum(np.power(inputs[:-n_fr_pred, output_ind] - rcn_targ, 2), axis=0)
                mse[:, 1] += np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)
                mse[:, 2] += np.sum(np.power(rcn_pred_l1_new - rcn_targ, 2), axis=0)
                mse[:, 3] += np.sum(np.power(rcn_targ, 2), axis=0)
                mse[:, 4] += np.sum(rcn_targ, axis=0)
                mse[:, 5] += rcn_targ.shape[0]

                mean_mse[idx, 0] = np.mean(np.sqrt(np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)) / sd)
                mean_mse[idx, 1] = np.mean(np.sqrt(np.sum(np.power(rcn_pred_l1_new - rcn_targ, 2), axis=0)) / sd)
                with open(main_data_path + 'eval/' + fld_names + '_output_' + str(
                        np.int32(raw_data['shotnum'][0])) + '.pkl',
                          'wb') as f:
                    pickle.dump(mse, f)
                np.savetxt(main_data_path + 'eval/batchendshotadapt_' + fld_names + '_output_' + str(
                    np.int32(raw_data['shotnum'][0])) + '.txt', np.hstack((rcn_targ, rcn_pred_l1, rcn_pred_l1_new)))
            # --------

        else:
            too_short += 1

    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]

    rmse_baseline = np.sqrt(mse[:, 0] / mse[:, 5])
    nrmse_baseline = rmse_baseline / sd

    rmse_rcn_l1 = np.sqrt(mse[:, 1] / mse[:, 5])
    nrmse_rcn_l1 = rmse_rcn_l1 / sd

    rmse_rcn_l2 = np.sqrt(mse[:, 2] / mse[:, 5])
    nrmse_rcn_l2 = rmse_rcn_l2 / sd

    print('%i out of %i (%.1f%%) were shorter than %i frames!' % (
        too_short, len(shotnum_list), too_short * 100.0 / len(shotnum_list), n_fr_pred))
    Rep = 'AvgNRMSE: Base,L1(%i),LinReg -> %.3e, %.3e, %.3e\n' % (
        W_res1.shape[0], np.mean(nrmse_baseline), np.mean(nrmse_rcn_l1), np.mean(nrmse_rcn_l2))
    print(Rep)
    print(tm)
    return mse, mean_mse, Rep


##############################################################################
def inside_adaptation(fld_names, main_data_path, n_fr_pred, input_ind, output_ind, frm_stck, cal_drv, nread, W_in1,
                      W_res1,
                      W_bi1, beta1, leak, beta_reg, xTx, xTy, xlen, regu):
    xTx_ad = np.zeros((W_res1.shape[0] + 1, W_res1.shape[0] + 1), dtype='float64')
    xTy_ad = np.zeros((W_res1.shape[0] + 1, nread), dtype='float64')
    xlen_ad = 0
    beta1_new = beta1.copy()
    adapt_ind = 0
    final_mse = []
    adapt_time = 10
    profiles = ['dens', 'temp', 'press_EFIT01', 'q_EFIT01', 'rotation']
    a = [os.remove(i) for i in glob.glob(main_data_path + 'eval/' + fld_names + '_*.pkl')]

    mse = np.zeros((325, 6))

    shotnum_list = [k.split('\\')[-1].split('.')[0] for k in sorted(glob.glob(main_data_path + fld_names + '/*.pkl'))]
    # if fld_names == 'test':
    #     lstm_shot_names = [k.split('\\')[-1].split('.')[0] for k in glob.glob('../BOF_data/LSTM_prediction/test/*.pkl')]
    #     shotnum_list = list(set(lstm_shot_names) & set(shotnum_list))
    #     # a = [shotnum_list.remove(k) for k in ['156520', '162720', '164280', '170590', '172380', '175970']]
    too_short = 0
    tm = np.zeros(2)
    mean_mse = np.zeros((len(shotnum_list), 2))
    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]
    # for idx in tqdm(range(len(shotnum_list) - 1), desc='Adapt', ascii=True,
    #                dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
    for idx in range(len(shotnum_list) - 1):
        tr_shot = shotnum_list[idx]
        vl_shot = shotnum_list[idx + 1]
        raw_data = pickle.load(open(main_data_path + fld_names + '/' + tr_shot + '.pkl', 'rb'))
        if (len(raw_data['shotnum']) > n_fr_pred):
            t1 = time.time()
            inputs = 0
            for keys, values in raw_data.items():
                if keys != 'shotnum':
                    if type(inputs) == int:
                        inputs = np.reshape(np.array(values), (np.array(values).shape[0], -1))
                    else:
                        inputs = np.hstack((inputs, np.reshape(np.array(values), (inputs.shape[0], -1))))

            inputs_tmp = inputs.copy()
            inputs_tmp[:-n_fr_pred, -4:] = inputs_tmp[n_fr_pred:, -4:]  # include actuators of the future
            u = calderiv(frame_stacking(inputs_tmp[:-n_fr_pred, input_ind], frm_stck), cal_drv)

            d = inputs[n_fr_pred:, output_ind] - inputs[:-n_fr_pred, output_ind]
            x1 = res_exe(W_in1, W_res1, W_bi1, leak[0], u)

            # ----split x,d
            chunks = 10
            x1_chunk = [x1[i:i + chunks] for i in range(0, x1.shape[0], chunks)]
            d_chunk = [d[i:i + chunks] for i in range(0, d.shape[0], chunks)]
            o1_new = []

            for curr_x, curr_d in zip(x1_chunk, d_chunk):
                t1 = time.time()
                o1_new.append(np.dot(curr_x, beta1_new))
                xlen_ad += curr_x.shape[0]
                xTx_ad += np.dot(curr_x.T, curr_x)
                xTy_ad += np.dot(curr_x.T, curr_d)
                beta1_new = res_train(xTx + xTx_ad, xTy + xTy_ad, xlen + xlen_ad, regu[0])
                tm[0] += (time.time() - t1)
                tm[1] += 1
            o1_new = np.vstack(o1_new)
            o1 = np.dot(x1, beta1)
            rcn_targ = inputs[n_fr_pred:, output_ind]
            rcn_pred_l1 = inputs[:-n_fr_pred, output_ind] + o1  # [:-n_fr_pred, :]
            rcn_pred_l1_new = inputs[:-n_fr_pred, output_ind] + o1_new  # [:-n_fr_pred, :]

            mse[:, 0] += np.sum(np.power(inputs[:-n_fr_pred, output_ind] - rcn_targ, 2), axis=0)
            mse[:, 1] += np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)
            mse[:, 2] += np.sum(np.power(rcn_pred_l1_new - rcn_targ, 2), axis=0)
            mse[:, 3] += np.sum(np.power(rcn_targ, 2), axis=0)
            mse[:, 4] += np.sum(rcn_targ, axis=0)
            mse[:, 5] += rcn_targ.shape[0]

            mean_mse[idx, 0] = np.mean(np.sqrt(np.sum(np.power(rcn_pred_l1 - rcn_targ, 2), axis=0)) / sd)
            mean_mse[idx, 1] = np.mean(np.sqrt(np.sum(np.power(rcn_pred_l1_new - rcn_targ, 2), axis=0)) / sd)
            with open(main_data_path + 'eval/' + fld_names + '_output_' + str(
                    np.int32(raw_data['shotnum'][0])) + '.pkl',
                      'wb') as f:
                pickle.dump(mse, f)
            np.savetxt(main_data_path + 'eval/inshotadapt200ms_' + fld_names + '_output_' + str(
                np.int32(raw_data['shotnum'][0])) + '.txt', np.hstack((rcn_targ, rcn_pred_l1, rcn_pred_l1_new)))
        # --------

        else:
            too_short += 1

    sd = np.loadtxt('../sd2_' + fld_names + '.txt')[:325]

    rmse_baseline = np.sqrt(mse[:, 0] / mse[:, 5])
    nrmse_baseline = rmse_baseline / sd

    rmse_rcn_l1 = np.sqrt(mse[:, 1] / mse[:, 5])
    nrmse_rcn_l1 = rmse_rcn_l1 / sd

    rmse_rcn_l2 = np.sqrt(mse[:, 2] / mse[:, 5])
    nrmse_rcn_l2 = rmse_rcn_l2 / sd

    print('%i out of %i (%.1f%%) were shorter than %i frames!' % (
        too_short, len(shotnum_list), too_short * 100.0 / len(shotnum_list), n_fr_pred))
    Rep = 'AvgNRMSE: Base,L1(%i),LinReg -> %.3e, %.3e, %.3e\n' % (
        W_res1.shape[0], np.mean(nrmse_baseline), np.mean(nrmse_rcn_l1), np.mean(nrmse_rcn_l2))
    print(Rep)
    print(tm)
    return mse, mean_mse, Rep


# --------------------------------------------------------------
#--------------------------------------------------------------
def read_file(fpath):
    tp=fpath[fpath.rfind('.')+1:]
    if tp=='08': # aurora file
        x=np.fromfile(fpath,dtype=np.int16)
        x.byteswap(True)
        fs=8000
    else: # wave file
        fs,x=scipy.io.wavfile.read(fpath)
    s=x/float(max(x))
    return s,fs

def mfcc_features(n_input, SpecSub=False, w='tMpedD', fl_ms=30, inc_ms=10, nc=12, p=24, cut_sl_flag=False, Vlen=4,
                  Alen=1, fl=0, fh=0.5):
    """
    # Inputs:
    #     x     speech signal
    #     fs  sample rate in Hz (default 8000)
    #     nc  number of cepstral coefficients excluding 0'th coefficient (default 12)
    #     fl_ms   length of frame (new default 30ms))
    #     inc_ms frame increment (default 10ms)
    #     p   number of filters in filterbank
    #     fl  low end of the lowest filter as a fraction of fs (default = 0)
    #     fh  high end of highest filter as a fraction of fs (default = 0.5)
    #
    #        w   any sensible combination of the following:
    #
    #                'N'    Hanning window in time domain
    #                'M'    Hamming window in time domain (default)
    #
    #              't'  triangular shaped filters in mel domain (default)
    #              'n'  hanning shaped filters in mel domain
    #              'm'  hamming shaped filters in mel domain
    #
    #                'p'    filters act in the power domain
    #                'a'    filters act in the absolute magnitude domain (default)
    #
    #               '0'  include 0'th order cepstral coefficient
    #                'e'  include log energy
    #                'd'    include delta coefficients (dc/dt)
    #                'D'    include delta-delta coefficients (d^2c/dt^2)
    #
    #              'z'  highest and lowest filters taper down to zero (default)
    #              'y'  lowest filter remains at 1 down to 0 frequency and
    #                     highest filter remains at 1 up to nyquist freqency
    """
    x, fs = read_file(n_input)
    frame_length = fs * fl_ms / 1000
    inc = fs * inc_ms / 1000  # Increment
    z_org = enframe(x, frame_length, inc, w)
    if cut_sl_flag:
        z, cut_beg, cut_end, WInput = cut_silence(z_org, fs, inc)
    else:
        z = z_org
    # f=scipy.fft(z.T).T # gives the FFT of input
    f = abs(scipy.fft(z.T)).T  # gives the FFT of input
    f = f[0:np.floor(frame_length / 2) + 1, :]
    if SpecSub:  # Spectral subtraction
        f_SS = Spectral_Subtraction(f)
    else:
        f_SS = f
    # --------- BEGIN Denoising DFT-------
    # f_SS=DAE_flow(calderiv(np.array(f_SS).T,4,1,w)).T
    # --------- END   Denoising DFT-------
    m, a, b = melbankm(p, frame_length, fs, fl, fh, w)
    pw = np.real(np.multiply(f_SS[a:b, :], np.conj(f_SS[a:b, :])))  # the energy of the input
    pth = np.max(pw) * 1E-20;
    if w.find('p') != -1:
        y = np.log(np.maximum(m * pw, pth));
    else:
        ath = np.sqrt(pth);
        y = np.log(np.maximum(m * abs(f_SS[a:b, :]), ath));
    # --------- BEGIN Denoising MelFB-------
    # y=DAE_flow[:-1](feature_Normalizing(calderiv(np.array(y).T,4,1,w))).T
    # y=y[:24,:]
    # --------- END   Denoising MelFB-------
    c = rdct(y).T  # DCT
    nf = c.shape[0]
    nc += 1
    if p > nc:
        c = c[:, 0:nc]
    elif p < nc:
        c = np.hstack((c, np.zeros(nf, nc - p)))
    if w.find('O') == -1:
        c = c[:, 1:]
        nc -= 1

    if w.find('e') != -1:
        c = np.hstack((np.resize(np.log(sum(pw)), (nf, 1)), c))
        nc += 1
    c = calderiv(c, 4, 1, w)
    y = calderiv(np.array(y.T), 4, 1, w)
    dft = calderiv(np.array(f_SS.T), 4, 1, w)
    return c, y


# --------------------------------------------------------------

def calderiv(c,Vlen=400,Alen=100,w='dD'):
    nf=c.shape[0]
    nc=c.shape[1]
    dv=denom_delta(Vlen)
    da=denom_delta(Alen)
    if w.find('D')!=-1:
        vf=np.array(range(Vlen,-(Vlen+1),-1))/dv
        af=np.array(range(Alen,-(Alen+1),-1))/da
        cx=np.vstack((np.tile(c[0],[Vlen+Alen,1]),c,np.tile(c[-1],[Vlen+Alen,1])))
        vx=np.reshape(scipy.signal.lfilter(vf,1,cx.flatten('F')),(nf+2*(Vlen+Alen),nc),order='F')
        vx=np.delete(vx,range(2*Vlen),0)
        ax=np.reshape(scipy.signal.lfilter(af,1,vx.flatten('F')),(nf+2*Alen,nc),order='F')
        ax=np.delete(ax,range(2*Alen),0)
        vx=vx[Alen:nf+Alen,:]
        if w.find('d')!=-1:
            c=np.hstack((c,vx,ax))
        else:
            c=np.hstack((c,ax))
    elif w.find('d')!=-1:
        vf=np.array(range(Vlen,-(Vlen+1),-1))/dv
        cx=np.vstack((np.tile(c[0],[Vlen,1]),c,np.tile(c[-1],[Vlen,1])))
        vx=np.reshape(scipy.signal.lfilter(vf,1,cx.flatten('F')),(nf+2*Vlen,nc),order='F')
        vx=np.delete(vx,range(2*Vlen),0)
        c=np.hstack((c,vx))
    return c


# --------------------------------------------------------------
def melfb(x, w, fs, SpecSub=False, fl_ms=30, inc_ms=10, p=40, cut_sl_flag=False, Vlen=4, Alen=1, fl=0, fh=0.5):
    nc = p
    frame_length = fs * fl_ms / 1000
    inc = fs * inc_ms / 1000  # Increment
    z_org = enframe(x, frame_length, inc, w)
    if cut_sl_flag:
        z, cut_beg, cut_end, WInput = cut_silence(z_org, fs, inc)
    else:
        z = z_org
    f = scipy.fft(z.T)  # gives the FFT of input
    f = f.T
    f = f[0:np.floor(frame_length / 2) + 1, :]
    if SpecSub:  # Spectral subtraction
        f_SS = Spectral_Subtraction(f)
    else:
        f_SS = f
    m, a, b = melbankm(p, frame_length, fs, fl, fh, w)
    pw = np.real(np.multiply(f_SS[a:b, :], np.conj(f_SS[a:b, :])))  # the energy of the input
    pth = np.max(pw) * 1E-20;
    if w.find('p') != -1:
        # c=np.log(np.maximum(m*pw,pth));
        c = np.power(np.maximum(m * pw, pth), 1.0 / 3);
    else:
        ath = np.sqrt(pth);
        # c=np.log(np.maximum(m*abs(f_SS[a:b,:]),ath));
        c = np.power(np.maximum(m * abs(f_SS[a:b, :]), ath), 1.0 / 3);
    c = c.T
    nf = c.shape[0]
    # nc+=1
    # c=c[:,0:nc]
    if w.find('e') != -1:
        c = np.hstack((np.resize(np.log(sum(pw)), (nf, 1)), c))
        nc += 1
    # Derivatives
    dv = denom_delta(Vlen)
    da = denom_delta(Alen)
    if w.find('D') != -1:
        vf = np.array(range(Vlen, -(Vlen + 1), -1)) / dv
        af = np.array(range(Alen, -(Alen + 1), -1)) / da
        cx = np.vstack((np.tile(c[0], [Vlen + Alen, 1]), c, np.tile(c[-1], [Vlen + Alen, 1])))
        vx = np.reshape(scipy.signal.lfilter(vf, 1, cx.flatten(1)), (nf + 2 * (Vlen + Alen), nc), order='F')
        vx = np.delete(vx, range(2 * Vlen), 0)
        ax = np.reshape(scipy.signal.lfilter(af, 1, vx.flatten(1)), (nf + 2 * Alen, nc), order='F')
        ax = np.delete(ax, range(2 * Alen), 0)
        vx = vx[Alen:nf + Alen, :]
        if w.find('d') != -1:
            c = np.hstack((c, vx, ax))
        else:
            c = np.hstack((c, ax))
    elif w.find('d') != -1:
        vf = np.array(range(Vlen, -(Vlen + 1), -1)) / dv
        cx = np.vstack((np.tile(c[0], [Vlen, 1]), c, np.tile(c[-1], [Vlen, 1])))
        vx = np.reshape(scipy.signal.lfilter(vf, 1, cx.flatten(1)), (nf + 2 * Vlen, nc), order='F')
        vx = np.delete(vx, range(2 * Vlen), 0)
        c = np.hstack((c, vx))
    return np.array(c[1:-1, :].T)  # JUST TO MATCH THE SIZE WITH THE ALIGNMENT!!!!


# --------------------------------------------------------------

def denom_delta(N):
    d = 0
    for i in range(N + 1):
        d += pow(i, 2)
    return float(2 * d)


# --------------------------------------------------------------

def enframe(x, fl, inc, w):
    if w.find('N') != -1:
        win = np.hanning(fl)
    else:
        win = np.hamming(fl)
    nx = x.shape[0];
    nf = int(np.floor((nx - fl + inc) / inc));
    f = np.zeros([nf, fl])
    indf = inc * np.array(range(0, nf));
    inds = np.array(range(0, fl));
    indf_t = []
    inds_t = []
    indf_t = np.tile(indf, [fl, 1])
    inds_t = np.tile(inds, [nf, 1])
    xindex = np.array(indf_t.T + inds_t)
    f = x[xindex]
    w = win.T;
    D = []
    for i in range(fl):
        D.append(w[i] * f[:, i])
    return np.array(D)


# --------------------------------------------------------------


def melbankm(p, n, fs, fl, fh, w='tz'):
    f0 = 700.0 / fs
    fn2 = np.int32(np.floor(n / 2.0))
    lr = np.log((f0 + fh) / (f0 + fl)) / (p + 1)
    # convert to fft bin numbers with 0 for DC term
    bl = n * ((f0 + fl) * np.exp(np.multiply([0, 1, p, p + 1], lr)) - f0)
    b2 = np.int32(np.ceil(bl[1]))
    b3 = np.int32(np.floor(bl[2]) + 1)
    if w.find('y') != -1:
        pf = np.log((f0 + np.array(range(b2, b3)) / np.float64(n)) / (f0 + fl)) / lr
        fp = np.floor(pf)
        r = np.hstack((np.zeros((1, b2)), [fp - 1], [fp], p - 1 * np.ones((1, fn2 - b3 + 1))))[0]
        c = np.hstack((np.array(range(0, b3)), np.array(range(b2, fn2 + 1))))
        v = 2 * np.hstack(([[0.5]], np.ones((1, b2 - 1)), [1 - pf + fp], [pf - fp], np.ones((1, fn2 - b3)), [[0.5]]))[0]
        mn = 0
        mx = fn2 + 1
    else:
        b1 = np.int32(np.floor(bl[0]) + 1)
        b4 = np.int32(np.minimum(fn2, np.ceil(bl[3])))
        pf = np.log((f0 + np.array(range(b1, b4)) / np.float64(n)) / (f0 + fl)) / lr
        fp = np.floor(pf)
        pm = pf - fp
        k2 = b2 - b1 + 1
        k3 = b3 - b1 + 1
        k4 = b4 - b1 + 1
        r = (np.hstack((fp[k2 - 1:k4], fp[0:k3 - 1] + 1))) - 1
        c = (np.hstack((np.array(range(k2, k4)), np.array(range(1, k3))))) - 1
        v = 2 * np.hstack((1 - pm[k2 - 1:k4], pm[0:k3 - 1]))
        mn = b1
        mx = b4
    if w.find('n') != -1:
        v = 1 - np.cos(v * np.pi / 2)
    elif w.find('m') != -1:
        v = 1 - 0.92 / 1.08 * np.cos(v * np.pi / 2)
    x = np.matrix(np.zeros((max(r) + 1, max(c) + 1)))
    for i in range(len(c)):
        x[int(r[i]), int(c[i])] = v[i]
    return x, mn, mx


# --------------------------------------------------------------


def rdct(x):
    fl = x.shape[0] == 1;
    m, k = x.shape;
    b = 1;
    a = np.sqrt(2 * m);
    x = np.vstack((x[range(0, m, 2), :], x[range(int(2 * np.floor(m / 2) - 1), 0, -2), :]));
    z = np.hstack((np.sqrt(2), 2 * np.exp((-0.5j * np.pi / m) * (np.array(range(m - 1)) + 1)))).T
    tmp = np.tile(z, [k, 1]).T
    y = np.real(scipy.fft(x.T).T * tmp) / a;
    if fl:
        y = y.T
    return y


# --------------------------------------------------------------

def cut_silence(z, fs, inc, alpha=0.6, beta=1e-1, smoothing_frame=3, energy_smooth_frame=5, removing_ms=250):
    weak_input = ''
    esm_ind = energy_smooth_frame / 2
    fr_per_s = fs / inc
    removing_fr = removing_ms * fr_per_s / 1000
    z_in = z[:, removing_fr:-removing_fr]  # first and last 100ms is removed for microphone instability
    N_fr = z_in.shape[1]
    Energy = sum(np.power(z_in, 2))
    smoothed_Energy = np.zeros(N_fr)
    for i in range(esm_ind, N_fr - esm_ind):
        smoothed_Energy[i] = sorted(Energy[i - esm_ind:i + esm_ind + 1])[esm_ind]
    smoothed_Energy[range(esm_ind)] = smoothed_Energy[esm_ind]
    smoothed_Energy[range(N_fr - esm_ind, N_fr)] = smoothed_Energy[N_fr - esm_ind - 1]
    if max(smoothed_Energy) < 5 * np.mean(smoothed_Energy[:20]):
        weak_input = " (Weak input)"
    Energy_threshold = (1 + alpha) * min(smoothed_Energy) + max(smoothed_Energy) * beta;
    sm_ind = smoothing_frame / 2
    sm_begin = smoothing_frame / 2
    while min(smoothed_Energy[sm_begin - sm_ind:sm_begin + sm_ind + 1]) < Energy_threshold:
        sm_begin += 1
        if sm_begin == N_fr - 1:
            print
            "No voice has been saved!!!"
            break
    sm_end = N_fr - sm_ind - 1
    while min(smoothed_Energy[sm_end - sm_ind:sm_end + sm_ind + 1]) < Energy_threshold:
        sm_end -= 1
        if sm_end == 0:
            print
            "No voice has been saved!!!"
            break
    begin_fram = max(sm_begin - fr_per_s / 2,
                     0)  # add 500 ms to the clipped part, to be sure that voice is not clipped
    end_frame = min(sm_end + fr_per_s / 2, N_fr)
    z_out = z_in[:, begin_fram:end_frame]
    return z_out, begin_fram, end_frame, weak_input


# --------------------------------------------------------------


def Spectral_Subtraction(f, bins=16, silence_frame_est=25, alpha=0.9):
    n_sp, n_fr = f.shape
    f = abs(f)
    log_f = np.log(f + 1e-8)
    hist_log_f = log_f
    thrshld = np.zeros([n_sp, 1])
    for i in range(n_sp):
        max_s = 0
        n_bin = 16
        max_sp = max(hist_log_f[i, :])
        hist_log_f[i, :] = np.maximum(hist_log_f[i, :], max_sp - 10)
        I, IX = myhist(hist_log_f[i, :], n_bin)
        j = 1
        while sum(I[0:j]) < silence_frame_est:
            j += 1
        thrshld = IX[j - 1]
        ind1 = np.nonzero(log_f[i, :] < thrshld)
        ind2 = np.nonzero(log_f[i, :] >= thrshld)
        log_f[i, ind1] = np.log(1 - alpha) + thrshld
        log_f[i, ind2] = log_f[i, ind2] + np.log(1 - alpha * np.exp(-(log_f[i, ind2] - thrshld)))

    f2 = np.exp(log_f)
    return f2


# --------------------------------------------------------------

def MA_smoothing(c, smoothing_length=3):
    nMfcc, nfr = c.shape
    smoothing_Index = smoothing_length / 2
    smoothing_Center = smoothing_length / 2
    Begin_smoothing = smoothing_Center
    End_smoothing = nfr - smoothing_Index
    smoothed_c = []
    for i in range(Begin_smoothing, End_smoothing):
        smoothed_c.append(np.mean(c[:, i - smoothing_Index:i + smoothing_Index + 1], axis=1))
    NoiseRobust_MFCC = np.array(smoothed_c).T
    NoiseRobust_MFCC = np.concatenate(
        (NoiseRobust_MFCC[:, :smoothing_Index], NoiseRobust_MFCC, NoiseRobust_MFCC[:, -smoothing_Index:]), 1)
    return NoiseRobust_MFCC


# --------------------------------------------------------------

def Group_Normalizing(sample, mean_c, rescale_f):
    nc, nfr = sample.shape
    norm_sample = np.zeros(sample.shape)
    norm_sample = (sample - mean_c[:nc]) * rescale_f[:nc]
    return norm_sample





# NEW--------

def preprocess_label(fname):
    df_labels = pd.read_csv(fname, header=0, sep=',')
    df_labels.drop(['egam'], axis=1, inplace=True)
    # The line below throws a ValueError, keys and values overlapping. 
    #df_labels.replace(
    #    {'baae': {-1:0 , 0:0 , 1:0, 2:0, 3:1, 4:0},
    #     'bae':   {-1:0 , 0:0 , 1:0, 2:1, 3:1, 4:0},
    #     'eae':   {-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0},
    #     'rsae':  {-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0},
    #     'tae':    {-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0}}, inplace=True)
    # This is te fix for the ValueError
    df_labels["baae"] = df_labels["baae"].map({-1:0 , 0:0 , 1:0, 2:0, 3:1, 4:0})
    df_labels["bae"] = df_labels["bae"].map({-1:0 , 0:0 , 1:0, 2:1, 3:1, 4:0})
    df_labels["eae"] = df_labels["eae"].map({-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0})
    df_labels["rsae"] = df_labels["rsae"].map({-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0})
    df_labels["tae"] = df_labels["tae"].map({-1:0 , 0:0 , 1:0, 2:1, 3:0, 4:0})
    df_labels = df_labels[(df_labels.baae > 0) | (df_labels.bae > 0) | (df_labels.eae > 0) | (df_labels.rsae >0) | (df_labels.tae >0)]
    return df_labels

def create_target(labels, inp, inp_type, m_win , fs = 500000,t=[]):
    ae_win=m_win.copy()
    labels = labels.copy(deep=True)
    ae_win = {x: np.int32(ae_win[x]/2) for x in ae_win}
    inp_len = inp.shape[0]
    if inp_type == 'ece':
        labels['time'] = labels['time'].values * fs/1000
        ae_win = {x: np.int32(ae_win[x]*fs/1000) for x in ae_win}
    target = np.zeros((inp_len,len(ae_win)))
    for index, row in labels.iterrows():
        for idx,key in enumerate(ae_win):
            wlen = ae_win[key]
            if row[key] !=0:
                if inp_type == 'ece':
                    beg = np.max([0, np.int32(row['time']-wlen)])
                    end = np.min([inp_len, np.int32(row['time']+wlen)])
                elif inp_type == 'spectra':
                    beg = np.max([0, (np.abs((t*1000) - row['time']+wlen)).argmin()])
                    end = np.min([inp_len, (np.abs((t*1000) - row['time']-wlen)).argmin()])
                else:
                    print('create_target - wrong input type' )
                
                target[beg:end,idx] = row[key]
    return (target)


def validation (X,y,S, win, wres, wbi, beta, leak):
    out=dict()
    Err = np.zeros((len(X),2+2*y[0].shape[1]))
    idx=0
    for u,d,s in tqdm(zip(X,y,S), desc='Evaluate ...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        x1 = rcfun.res_exe(win, wres, wbi, leak, u)
        o1 = np.dot(x1,beta)
        Err[idx,:]=np.hstack([int(s),u.shape[0],np.sum(np.power((o1-d),2),axis=0),np.sum(d,axis=0)])
        out[int(s)]=np.hstack((d,o1))
        idx+=1
    return Err,out
    
def validation_l2 (X,y,S, win, wres, wbi, beta, leak):
    out=dict()
    Err = np.zeros((len(X),2+3*y[0].shape[1]))
    idx=0
    x = [None] * 2
    o = [None] * 2
    for u,d,s in tqdm(zip(X,y,S), desc='Evaluate ...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        x[0] = rcfun.res_exe(win[0], wres[0], wbi[0], leak[0], u)
        o[0] = np.dot(x[0],beta[0])
        x[1] = rcfun.res_exe(win[1], wres[1], wbi[1], leak[1], o[0])
        o[1] = np.dot(x[1],beta[1])
        Err[idx,:]=np.hstack([int(s),u.shape[0],np.sum(np.power((o[0]-d),2),axis=0),np.sum(np.power((o[1]-d),2),axis=0),np.sum(d,axis=0)])
        out[int(s)]=np.hstack((d,o[0],o[1]))
        idx+=1
    return Err,out

def shot_spec(shotn,ecen,ax,cut_shot,datapath,mode_win,mode_color,random_list_plot,spec_params,vmin=100,vmax=200):
    all_labels = preprocess_label('uci_labels.txt')
    ece_data = pickle.load(open(datapath+'ece_'+str(shotn)+'.pkl','rb'))
    shot_label = all_labels[all_labels['shot']==shotn]
    plot_marks = shot_label.apply(lambda row: row[row == 1].index, axis=1).values
    ece_num = '\\tecef%.2i' % (ecen)
    
    sig_in = ece_data[ece_num][:np.int32(cut_shot*spec_params['fs'])]
    f, t, Sxx = scipy.signal.spectrogram(sig_in, nperseg=spec_params['nperseg'], noverlap=spec_params['noverlap'],fs=spec_params['fs'], window=spec_params['window'],scaling=spec_params['scaling'], detrend=spec_params['detrend'])
    img = np.flipud(np.log(Sxx + spec_params['eps']))
    gray=(255-255*(img-np.min(img))/(np.max(img)-np.min(img))).astype('uint8')
    gray[gray<vmin]=vmin
    gray[gray>vmax]=vmax
    c=ax.pcolormesh(t*1000, f[::-1]/1000, gray,shading='gouraud',cmap='magma_r')
    _=plt.colorbar(c,ax=ax)
    _=plt.xlabel('Time (ms)')
    _=plt.ylabel('Freq. (KHz) -- '+ece_num[2:])
    for tm,mrk in zip(shot_label['time'],plot_marks):
        top_loc_id=random.choice(random_list_plot)
        ax.plot([tm,tm],[0,spec_params['fs']/1000*top_loc_id],'w--')
        for i,m in enumerate(mrk):
            ax.add_patch(
                mpatches.Rectangle((tm-(mode_win[m]/2), 0), mode_win[m], spec_params['fs']/1000*top_loc_id, ec=mode_color[m], fc='none',lw=2,ls='-'))
            ax.annotate(m, (tm, spec_params['fs']/1000*(top_loc_id+0.02)+(i*10)), color=mode_color[m], weight='bold', 
                        fontsize=10, ha='center', va='center')
    return (f, t, Sxx)



def create_probabilities(y_org,priors,lookup_t=[],eps=0.001,beta=0.01):
    y=y_org.copy()

    if len(lookup_t)==0:
        #y+=1
        max_y=np.max(y)
        y=y/max_y
    y=np.maximum(y,eps)
    p_x_h=np.log(y/(priors+beta))
    return p_x_h

def create_ece_dataset (shotlist,dpath, fs, samp_rate, mode_win, cut_shot, derv, norm,fr_stack,vlen ,alen):
    output = dict()
    all_ece_file_names = glob.glob(dpath+'ece_*.pkl')
    all_labels = preprocess_label('uci_labels.txt')
#     shotlist= all_labels.shot.unique()[rng]
    output = {s:dict() for s in shotlist}
    for shotn in tqdm(shotlist, desc='Data prep...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        fname = dpath+'ece_'+str(shotn)+'.pkl'
        if fname in all_ece_file_names:
            try:
                data = pickle.load(open(fname,'rb'))
            except:
                print('Broken -> '+fname)
                continue
            curr_x = np.vstack([ data['\\tecef%.2i' % (i+1)] for i in range(40) ]).T
            curr_x = curr_x[:np.int32(cut_shot*fs),:]
            curr_label = all_labels[all_labels['shot']==shotn]
            curr_y = create_target(curr_label, curr_x, 'ece', mode_win, fs)
            curr_x = curr_x[::samp_rate,:]
            curr_y = curr_y[::samp_rate,:]

            if derv:
                curr_x = calderiv(curr_x, vlen, alen, 'dD')
            if norm:
                stats = pickle.load(open('stats_ECEdD_mn_mx_avg_std.pkl','rb'))
                if derv:
                    mn = stats[2]
                    st = stats[3]
                else:
                    mn = np.mean(stats[2][:curr_x.shape[1]])
                    st = np.mean(stats[3][:curr_x.shape[1]])
#                     mn = np.mean(curr_x,axis=0)
#                     st = np.std(curr_x,axis=0)                
                curr_x = (curr_x-mn)/st
            curr_x = frame_stacking (curr_x,fr_stack)
            output[shotn]['X'] = curr_x
            output[shotn]['y'] = curr_y
        else:
            print(fname)
    return output
    
def create_dataset_spectra (rng, dpath, spec_params, mode_win, cut_shot, derv, norm,fr_stack,vlen ,alen):
    all_ece_file_names = glob.glob(dpath+'ece_*.pkl')
    all_labels = preprocess_label('uci_labels.txt')
    shotlist= all_labels.shot.unique()
    S = []
    X = []
    y = []
    for shotn in tqdm(shotlist[rng], desc='Data prep...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        fname = dpath+'ece_'+str(shotn)+'.pkl'
        if fname in all_ece_file_names:
            try:
                data = pickle.load(open(fname,'rb'))
            except:
                print('Broken -> '+fname)
                continue
            S.append(shotn)
            curr_x = []
            for idx in range(40):
                sig_in = data['\\tecef%.2i' % (idx+1)][:np.int32(cut_shot*spec_params['fs'])]
                f, t, Sxx = scipy.signal.spectrogram(sig_in, nperseg=spec_params['nperseg'], noverlap=spec_params['noverlap'],fs=spec_params['fs'], window=spec_params['window'],scaling=spec_params['scaling'], detrend=spec_params['detrend'])
                Sxx = np.log(Sxx + spec_params['eps'])#[:np.int32(spec_params['nperseg']/4),:]
                curr_x.append(Sxx) 
            curr_x = np.vstack(curr_x).T
            curr_label = all_labels[all_labels['shot']==shotn]
            curr_y = create_target(curr_label, curr_x, 'spectra', mode_win, spec_params['fs'],t)
            if derv:
                curr_x = calderiv(curr_x, vlen, alen, 'dD')
            if norm:
                mn = np.mean(curr_x,axis=0)
                st = np.std(curr_x,axis=0)                
                curr_x = (curr_x-mn)/st
            curr_x = frame_stacking (curr_x,fr_stack)
            X.append(curr_x)
            y.append(curr_y)
            # pickle.dump([curr_x,curr_y],open('/scratch/gpfs/aj17/ece_spectra/spectrum'+ fname[fname.rfind('_'):],'wb'))


        else:
            print(fname)
    return S, X, y
    
def create_state_map(st_dg):
    N=len(st_dg)
    state_map=np.int32(np.zeros((N,5)))
    state_map[:,0]=np.array(range(N))
    state_map[:,1]=st_dg
    state_map[:,4]=1
    ind=0
    for i in range(N):
        state_map[i,2]=ind
        state_map[i,3]=ind+state_map[i,1]
        ind=ind+state_map[i,1]
    frst_sil_st=state_map[-1,2]
    lst_sil_st=state_map[-1,3]
    tot_num_st=sum(state_map[:,1])
    return state_map
    
def nonzero_intervals(orgvec):
    vec=orgvec.copy()
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec)==0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    edges, = np.nonzero(np.diff((vec==0)*1))
    edge_vec = [edges+1]
    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])

def split(a, n):
    a=np.arange(a[0],a[1])
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_multistate_target (curr_target,state_map):  
    tot_num_st=sum(state_map[:,1])
    slen = curr_target.shape[0]
    pre_align = np.zeros(slen)

    new_target = np.zeros((slen,tot_num_st))

    for curr_class in range(curr_target.shape[1]):
        state_ind = np.arange(state_map[curr_class,2],state_map[curr_class,3])
        intervals=nonzero_intervals(curr_target[:,curr_class])
        for curr_interval in intervals:
            labels = split(curr_interval, state_map[curr_class,1])
            for rng, col_ind in zip(labels,state_ind):
                new_target[rng,col_ind]=1
    return new_target
    
def create_ece_b_nrms_dataset (rng, ece_dpath, b_dpath, nrms_dpath, fs_ece, samp_rate, mode_win, cut_shot, derv, norm,fr_stack,vlen ,alen):
    all_ece_file_names = glob.glob(ece_dpath+'ece_*.pkl')
    all_b_file_names = glob.glob(b_dpath+'b_*.h5')
    all_nrms_file_names = glob.glob(nrms_dpath+'nrms_*.h5')
    
    all_labels = preprocess_label('uci_labels.txt')
    shotlist_with_labels= all_labels.shot.unique()
    shotlist_ece = [int(s[s.rfind('_')+1:-4]) for s in all_ece_file_names]
    shotlist_b = [int(s[s.rfind('_')+1:-3]) for s in all_b_file_names]
    shotlist_nrms = [int(s[s.rfind('_')+1:-3]) for s in all_nrms_file_names]
    final_shotlist = list(set(shotlist_with_labels) & set(shotlist_ece) & set(shotlist_b) & set(shotlist_nrms))
    final_shotlist.sort()
    final_shotlist = np.array(final_shotlist)
    # print(len(final_shotlist))
    S = []
    X_ece = []
    X_b = []
    X_nrms = []
    y = []
    for shotn in tqdm(final_shotlist[rng], desc='Data prep...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        fname_ece = ece_dpath+'ece_'+str(shotn)+'.pkl'
        fname_b = b_dpath+'b_'+str(shotn)+'.h5'
        fname_nrms = nrms_dpath+'nrms_'+str(shotn)+'.h5'
        
        if (fname_ece in all_ece_file_names) and (fname_b in all_b_file_names):
           #---- reading ECE ----
            try:
                data_ece = pickle.load(open(fname_ece,'rb'))
            except:
                print('Broken -> '+fname_ece)
                continue
            S.append(shotn)
            curr_x_ece = np.vstack([ data_ece['\\tecef%.2i' % (i+1)] for i in range(40) ]).T
            if np.mean(curr_x_ece)>100: # ECEs of a few shots have been saved in ev, not in kev
                curr_x_ece/=1000
            curr_x_ece = curr_x_ece[:np.int32(cut_shot*fs_ece),:]
            curr_label = all_labels[all_labels['shot']==shotn]
            curr_y = create_target(curr_label, curr_x_ece, 'ece', mode_win, fs_ece)
            ece_interp = interp.interp1d(np.arange(curr_x_ece.shape[0]),curr_x_ece,axis=0,kind='linear')
            curr_x_ece = ece_interp(np.arange(0,curr_x_ece.shape[0],samp_rate))
            # curr_x_ece = curr_x_ece[::samp_rate,:]
            
            y_interp = interp.interp1d(np.arange(curr_y.shape[0]),curr_y,axis=0)
            curr_y = y_interp(np.arange(0,curr_y.shape[0],samp_rate))
            # curr_y = curr_y[::samp_rate,:]

            #---- reading B ----
            hf = h5py.File(fname_b, 'r')
            curr_x_b=[np.array(hf.get(k)) for k in hf.keys()]
            time_b = np.array(hf.get('time'))
            hf.close()
            fs_b = np.rint(1000/(time_b[1:]-time_b[:-1]).mean())            
            curr_x_b=np.vstack(curr_x_b)[:-1,:].T
            curr_x_b = curr_x_b [:(cut_shot*fs_b).astype(int),:]
            mn_b = np.mean(curr_x_b,axis=0)
            std_b = np.std(curr_x_b,axis=0)
            b_interp = interp.interp1d(np.arange(curr_x_b.shape[0]),curr_x_b,axis=0,kind='linear')
            curr_x_b = b_interp(np.linspace(0,curr_x_b.shape[0]-1,curr_x_ece.shape[0]))

            #---- reading NRMS ----

            hf = h5py.File(fname_nrms, 'r')
            curr_x_nrms=[np.array(hf.get(k)) for k in hf.keys()]
            time_nrms = np.array(hf.get('time'))
            hf.close()
            fs_nrms = np.rint(1000/(time_nrms[1:]-time_nrms[:-1]).mean())            
            curr_x_nrms=np.vstack(curr_x_nrms)[:-1,:].T
            curr_x_nrms = curr_x_nrms [:(cut_shot*fs_nrms).astype(int),:]
            mn_nrms = np.mean(curr_x_nrms,axis=0)
            std_nrms = np.std(curr_x_nrms,axis=0)
            nrms_interp = interp.interp1d(np.arange(curr_x_nrms.shape[0]),curr_x_nrms,axis=0,kind='linear')
            curr_x_nrms = nrms_interp(np.linspace(0,curr_x_nrms.shape[0]-1,curr_x_ece.shape[0]))

            if derv:
                curr_x_ece = calderiv(curr_x_ece, vlen, alen, 'dD')
                curr_x_b = calderiv(curr_x_b, vlen, alen, 'dD')
                curr_x_nrms = calderiv(curr_x_nrms, vlen, alen, 'dD')
                
            if norm:
                stats_ece = pickle.load(open('stats_ECEdD_mn_mx_avg_std.pkl','rb'))
                mn_ece = np.mean(stats_ece[2][:curr_x_ece.shape[1]])
                st_ece = np.mean(stats_ece[3][:curr_x_ece.shape[1]])
                # mn_ece = np.mean(curr_x_ece,axis=0)
                # st_ece = np.std(curr_x_ece,axis=0)                
                curr_x_ece = (curr_x_ece-mn_ece)/st_ece
                curr_x_b = (curr_x_b-np.mean(curr_x_b,axis=0))/np.std(curr_x_b,axis=0)
                curr_x_nrms = (curr_x_nrms-np.mean(curr_x_nrms,axis=0))/np.std(curr_x_nrms,axis=0)
            curr_x_ece = frame_stacking (curr_x_ece,fr_stack)
            curr_x_b = frame_stacking (curr_x_b,fr_stack)
            curr_x_nrms = frame_stacking (curr_x_nrms,fr_stack)
            
            
            X_ece.append(curr_x_ece)
            X_b.append(curr_x_b)
            X_nrms.append(curr_x_nrms)
            y.append(curr_y)
        else:
            print(fname_ece)
    return S, X_ece, X_b, X_nrms, y



def create_ece_beam_dataset (rng, ece_dpath, beam_dpath, fs_ece, samp_rate, mode_win, cut_shot, derv, norm,fr_stack,vlen ,alen):
    all_ece_file_names = glob.glob(ece_dpath+'ece_*.pkl')
    all_beam_file_names = glob.glob(beam_dpath+'pinj*.h5')
    
    
    all_labels = preprocess_label('uci_labels.txt')
    shotlist_with_labels= all_labels.shot.unique()
    shotlist_ece = [int(s[s.rfind('_')+1:-4]) for s in all_ece_file_names]
    shotlist_b = [int(s[s.rfind('_')+1:-3]) for s in all_beam_file_names]

    final_shotlist = list(set(shotlist_with_labels) & set(shotlist_ece) & set(shotlist_b))
    final_shotlist.sort()
    final_shotlist = np.array(final_shotlist)
    print(len(final_shotlist))
    S = []
    X_ece = []
    X_b = []
    X_nrms = []
    y = []
    for shotn in tqdm(final_shotlist[rng], desc='Data prep...', ascii=True, dynamic_ncols=True):  # mdp.utils.progress_bar(tr_file_names):
        fname_ece = ece_dpath+'ece_'+str(shotn)+'.pkl'
        fname_b = beam_dpath+'pinjf_'+str(shotn)+'.h5'
        print(fname_b)
        if (fname_ece in all_ece_file_names) and (fname_b in all_beam_file_names):
           #---- reading ECE ----
            try:
                data_ece = pickle.load(open(fname_ece,'rb'))
            except:
                print('Broken -> '+fname_ece)
                continue
            S.append(shotn)
            curr_x_ece = np.vstack([ data_ece['\\tecef%.2i' % (i+1)] for i in range(40) ]).T
            if np.mean(curr_x_ece)>100: # ECEs of a few shots have been saved in ev, not in kev
                curr_x_ece/=1000
            curr_x_ece = curr_x_ece[:np.int32(cut_shot*fs_ece),:]
            curr_label = all_labels[all_labels['shot']==shotn]
            curr_y = create_target(curr_label, curr_x_ece, 'ece', mode_win, fs_ece)
            ece_interp = interp.interp1d(np.arange(curr_x_ece.shape[0]),curr_x_ece,axis=0,kind='linear')
            curr_x_ece = ece_interp(np.arange(0,curr_x_ece.shape[0],samp_rate))
            # curr_x_ece = curr_x_ece[::samp_rate,:]
            
            y_interp = interp.interp1d(np.arange(curr_y.shape[0]),curr_y,axis=0)
            curr_y = y_interp(np.arange(0,curr_y.shape[0],samp_rate))
            # curr_y = curr_y[::samp_rate,:]

            #---- reading B ----
            hf = h5py.File(fname_b, 'r')
            curr_x_b=[np.array(hf.get(k)) for k in hf.keys()]
            time_b = np.array(hf.get('time'))
            hf.close()
            fs_b = np.rint(1000/(time_b[1:]-time_b[:-1]).mean())            
            curr_x_b=np.vstack(curr_x_b)[:-1,:].T
            curr_x_b = curr_x_b [:(cut_shot*fs_b).astype(int),:]
            mn_b = np.mean(curr_x_b,axis=0)
            std_b = np.std(curr_x_b,axis=0)
            b_interp = interp.interp1d(np.arange(curr_x_b.shape[0]),curr_x_b,axis=0,kind='linear')
            curr_x_b = b_interp(np.linspace(0,curr_x_b.shape[0]-1,curr_x_ece.shape[0]))


            if derv:
                curr_x_ece = calderiv(curr_x_ece, vlen, alen, 'dD')
                curr_x_b = calderiv(curr_x_b, vlen, alen, 'dD')
                
            if norm:
                stats_ece = pickle.load(open('stats_ECEdD_mn_mx_avg_std.pkl','rb'))
                mn_ece = np.mean(stats_ece[2][:curr_x_ece.shape[1]])
                st_ece = np.mean(stats_ece[3][:curr_x_ece.shape[1]])
                # mn_ece = np.mean(curr_x_ece,axis=0)
                # st_ece = np.std(curr_x_ece,axis=0)                
                curr_x_ece = (curr_x_ece-mn_ece)/st_ece
                curr_x_b = (curr_x_b-np.mean(curr_x_b,axis=0))/np.std(curr_x_b,axis=0)

            curr_x_ece = frame_stacking (curr_x_ece,fr_stack)
            curr_x_b = frame_stacking (curr_x_b,fr_stack)
            
            
            X_ece.append(curr_x_ece)
            X_b.append(curr_x_b)
            y.append(curr_y)
        else:
            print(fname_ece)
    return S, X_ece, X_b, y


def customized_confusion_mat (y_pred,thr,shotn,mode_win,samp_rate,fs):
    all_labels = preprocess_label('uci_labels.txt')
    labels = all_labels[all_labels['shot']==shotn]
    y_target=create_target(labels, np.zeros((samp_rate*y_pred.shape[0],1)), 'ece', mode_win, fs)
    y_interp = interp.interp1d(np.arange(y_target.shape[0]),y_target,axis=0)
    y_target = y_interp(np.arange(0,y_target.shape[0],samp_rate))
    y_pred=np.where(y_pred<thr,0,1)
    labels['time'] = labels['time'].values * fs/(1000*samp_rate)

    mode_win = {x: np.int32(mode_win[x]*fs/(2000*samp_rate)) for x in mode_win}
    inp_len = y_pred.shape[0]*samp_rate
    TP = np.zeros(5)
    P = np.zeros(5)
    for index, row in labels.iterrows():
        for idx,key in enumerate(mode_win):
            wlen = mode_win[key]
            if row[key] !=0:
                beg = np.max([0, np.int32(row['time']-wlen)])
                end = np.min([inp_len, np.int32(row['time']+wlen)])
                if np.sum(y_pred[beg:end,idx])>10:#0.1*(end-beg):
                    TP[idx]+=1
                P[idx]+=1
    TN = np.sum((y_target+y_pred)==0,axis=0)
    N = np.sum(y_target==0,axis=0)
    FN = P-TP
    FP = N-TN
    return TP,FP,TN,FN,y_pred,y_target

def metrics (conf_mat):
    ae_list = list(conf_mat.keys())
    total = dict.fromkeys(['tp','fp','tn','fn'])
    for met in ['tp','fp','tn','fn']:
        total[met] = sum([x[met] for x in conf_mat.values()])
    conf_mat['total'] = total
    for ae, conf in conf_mat.items():
        TP = conf['tp']
        FP = conf['fp']
        TN = conf['tn']
        FN = conf['fn']
        conf_mat[ae]['tpr'] = TP/(TP+FN)
        conf_mat[ae]['fpr'] = FP/(FP+TN)
        conf_mat[ae]['p'] = TP/(TP+FP)
        conf_mat[ae]['r'] = TP/(TP+FN)
        conf_mat[ae]['f1'] = 2*conf_mat[ae]['p']*conf_mat[ae]['r']/(conf_mat[ae]['p']+conf_mat[ae]['r'])
        conf_mat[ae]['ba'] = 0.5*(conf_mat[ae]['tpr']+1-conf_mat[ae]['fpr'])
    
    # Make the text format
    
    metr = list(conf_mat[ae_list[0]].keys())
    txt='\t'+'\t'.join(metr) + '\n'
    for ae,values in conf_mat.items():
        txt+=ae
        lst = list(values.values())
        for item in lst:
            txt+='\t%.3f' % (item)
        txt+='\n'
    return conf_mat,txt

def find_nearest(myarray, myvalue):
    myarray = np.asarray(myarray)
    idx = (np.abs(myarray - myvalue)).argmin()
    return idx,myarray[idx]

def calculate_specenergy(spect,hop):
    energy = dict()
    beg_ind = 0
    end_ind = beg_ind + hop
    while end_ind < spect.shape[1]:
        energy[beg_ind]=np.sum(spect[:,beg_ind:end_ind],axis=1)
        end_ind +=hop
        beg_ind +=hop

    return energy

# ---- Spectrogram enahncement begin ----

def specgr (sig_in,spec_params,thr=0.9, gaussblr_win=(31,3)):
    f, t, Sxx = scipy.signal.spectrogram(sig_in, nperseg=spec_params['nperseg'], noverlap=spec_params['noverlap'],fs=spec_params['fs'], window=spec_params['window'],scaling=spec_params['scaling'], detrend=spec_params['detrend'])
    Sxx = np.log(Sxx + spec_params['eps'])
    Sxx=(Sxx-np.min(Sxx))/(np.max(Sxx)-np.min(Sxx))
    Sxx = Sxx[:-1,:];f=f[:-1]
    
    Sxx_enhanced= quantfilt(Sxx,thr)
    Sxx_enhanced =  gaussblr(Sxx_enhanced,gaussblr_win)
    Sxx_enhanced = meansub(Sxx_enhanced)    
    Sxx_enhanced = morph(Sxx_enhanced)
    Sxx_enhanced = meansub(Sxx_enhanced)    
    return Sxx,Sxx_enhanced,f,t



def norm(data):
    mn = data.mean()
    std = data.std()
    return((data-mn)/std)

def rescale(data):
    return (data-data.min())/(data.max()-data.min())

def quantfilt(src,thr=0.9):
    filt = np.quantile(src,thr,axis=0)
    out = np.where(src<filt,0,src)
    return out

# gaussian filtering
def gaussblr(src,filt=(31, 3)):
    src = (rescale(src)*255).astype('uint8')
    out = cv2.GaussianBlur(src,filt,0)
    return rescale(out)

# mean filtering
def meansub(src):
    mn = np.mean(src,axis=1)[:,np.newaxis]
    out = np.absolute(src - mn)
    return rescale(out)

# morphological filtering
def morph(src):
    src = (rescale(src)*255).astype('uint8')
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    mask = cv2.morphologyEx(src, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    return rescale(mask)
# ---- Spectrogram enahncement end ----

def rcn_infer(w_in,w_res,w_bi,w_out,leak,r_prev,u):
    if scipy.sparse.issparse(w_in): # applying input weights to the input. Sparse and dense matrix multiplication is different in Python 
        a1 = w_in * u 
    else:
        a1=np.dot(w_in, u)
    a2 = w_res * r_prev # applying recurrent weights to the previous reservoir states
    r_now = np.tanh(a1 + a2 + w_bi) # adding bias and applying activation function
    r_now = (1 - leak) * r_prev + leak * r_now # applying leak rate
    y = np.dot(np.append([1],r_now),w_out) # applying the output weight
    return r_now,y



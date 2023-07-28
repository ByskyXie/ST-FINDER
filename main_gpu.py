import random

from model import ST_Finder
import torch
import numpy as np
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from early_stopping import EarlyStopping
import config
from adapter import MatrixDataset
from config import ModelConfigAbilene as configAb, ModelConfigGEANT as configGT


device = torch.device('cuda:0')
save_path = ".\\"  # current dir


def losses_interpolation(pred, gt):
    return F.l1_loss(pred, gt)

def losses_recovery(pred, gt):
    source, target = pred.reshape(-1), gt.reshape(-1)
    not_zero_pos = get_not_zero_position(target)
    s = source * not_zero_pos
    t = target
    return torch.mean(F.l1_loss(s, t))


def get_not_zero_position(inputs):
    return torch.clamp(torch.clamp(torch.abs(inputs), 0, 1e-32) * 1e36, 0, 1)


def NMAE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.abs(source - target)
    denominator = torch.abs(target)
    not_zero_pos = get_not_zero_position(target)
    return torch.sum(not_zero_pos * molecular) / torch.sum(denominator)


def error_rate(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.pow(source - target, 2)
    denominator = torch.pow(target, 2)
    not_zero_pos = get_not_zero_position(target)
    return torch.pow(torch.sum(not_zero_pos * molecular) / torch.sum(denominator), 1 / 2)


def get_all_metrices(recov_mat, recov_valid, recov_mat_missing, interpo_mat, interpo_valid):
    """ metric """
    zero_pos = (1 - get_not_zero_position(recov_mat_missing)) * get_not_zero_position(recov_valid)
    recov_mat *= zero_pos
    recov_valid *= zero_pos
    all_mat = torch.concat([recov_mat, interpo_mat], dim=0)
    all_valid = torch.concat([recov_valid, interpo_valid], dim=0)
    assert all_mat.shape == all_valid.shape

    avgER, avgNMAE, avgRMSE = 0, 0, 0
    for i in range(all_mat.shape[0]):
        avgER += error_rate(all_mat[i], all_valid[i])
        avgNMAE += NMAE(all_mat[i], all_valid[i])
        avgRMSE += torch.sqrt(F.mse_loss(all_mat[i], all_valid[i]))
    avgER, avgNMAE, avgRMSE  = avgER/all_mat.shape[0], avgNMAE/all_mat.shape[0], avgRMSE/all_mat.shape[0]
    return avgER, avgNMAE, avgRMSE


def train_model(config, model_cfg):
    early_stopping = EarlyStopping(save_path)
    # stFinder = torch.load('ST_Finder.pth')  # load model

    # TODO:Abilene, GEANT
    stFinder = ST_Finder(config, model_cfg).to(device)
    datasets = MatrixDataset(path=model_cfg.path, all_batch_num=model_cfg.all_batch_num,
                             predict_matrix_num=config.predict_matrix_num, input_matrix_num=config.input_matrix_num,
                             fn_get_traffic_matrix=model_cfg.fn_get_traffic_matrix, gpu_mode=True,
                             sampling_rate1=config.sampling_rate1, sampling_rate2=config.sampling_rate2,
                             scaler=model_cfg.scaler, using_tube_samping=config.using_tube_samping)

    # TODO:loader
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    trainloader = DataLoader(trains, batch_size=config.batch_size, shuffle=True, drop_last=True)
    developloader = DataLoader(develops, batch_size=config.batch_size, shuffle=True, drop_last=True)
    testsloader = DataLoader(tests, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # define optimizer
    optimer = torch.optim.Adam(stFinder.parameters(), lr=model_cfg.lr, betas=config.betas)  # betas=(0.9, 0.999)
    # training model
    for epoch in range(config.epochs):
        beg_time = time.time()
        avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
        for train, valid, time_seq, target_times in trainloader:
            pred = stFinder(train, time_seq, target_times)
            # print(pred.shape, valid.shape, train.shape)  # [8, 2, 1, 12, 12] [8, 1, 2, 12, 12] [8, 1, 8, 12, 12]

            recov_mat = torch.squeeze(pred[:, :1])  # recover the input mat at left middle
            recov_valid = torch.squeeze(valid[:, :, :1])

            interpo_mat = torch.squeeze(pred[:, 1:])  # interpolate matrix
            interpo_valid = torch.squeeze(valid[:, :, 1:])

            loss = losses_recovery(recov_mat, recov_valid) + losses_interpolation(interpo_mat, interpo_valid)

            optimer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stFinder.parameters(), model_cfg.max_grad)

            optimer.step()

            # compute accuracy
            pred, valid = pred/datasets.scaler, valid/datasets.scaler  # scaler is to abilene, the very small value of abilene make pred is all 0
            recov_mat_missing = (train[train.shape[2] // 2 - 1] / datasets.scaler).view(config.batch_size, -1)
            recov_mat = (pred[:, :1]).view(config.batch_size*config.predict_matrix_num, -1)  # recover the input mat at left middle
            recov_valid = (valid[:, :, :1]).view(config.batch_size*config.predict_matrix_num, -1)
            interpo_mat = (pred[:, 1:]).view(config.batch_size*config.predict_matrix_num, -1)  # interpolate matrix
            interpo_valid = (valid[:, :, 1:]).view(config.batch_size*config.predict_matrix_num, -1)

            avg_loss += loss.detach()
            aer, anmae, armse = get_all_metrices(recov_mat, recov_valid, recov_mat_missing, interpo_mat, interpo_valid)
            avg_er += aer.detach()
            avg_nmae += anmae.detach()
            avg_rmse += armse.detach()
            counter += 1

        print(f'Epoch:{epoch}\tloss=\t{avg_loss / counter}\ttrain_ER=\t{avg_er / counter}'
              f'\ttrain_NMAE=\t{avg_nmae / counter}\ttrain_RMSE=\t{avg_rmse / counter}', end='\t')

        # develop
        avg_loss, avg_er, avg_nmae, counter = 0, 0, 0, 0
        for develop, valid, time_seq, target_time in developloader:
            pred = stFinder(develop, time_seq, target_time)

            # compute accuracy
            pred, valid = pred / datasets.scaler, valid / datasets.scaler  # scaler is to abilene, the very small value of abilene make pred is all 0
            recov_mat_missing = (develop[develop.shape[2] // 2 - 1] / datasets.scaler).view(config.batch_size, -1)
            recov_mat = (pred[:, :1]).view(config.batch_size * config.predict_matrix_num, -1)  # recover the input mat at left middle
            recov_valid = (valid[:, :, :1]).view(config.batch_size * config.predict_matrix_num, -1)
            interpo_mat = (pred[:, 1:]).view(config.batch_size * config.predict_matrix_num, -1)  # interpolate matrix
            interpo_valid = (valid[:, :, 1:]).view(config.batch_size * config.predict_matrix_num, -1)

            aer, anmae, armse = get_all_metrices(recov_mat, recov_valid, recov_mat_missing, interpo_mat, interpo_valid)
            avg_er += aer.detach()
            avg_nmae += anmae.detach()
            avg_rmse += armse.detach()
            counter += 1
        print(f'Develop_ER=\t{avg_er / counter}\tDevelop_NMAE=\t{avg_nmae / counter}\tDevelop_RMSE=\t{avg_rmse / counter}\tTime:{time.time()-beg_time}')
        # Early stopping setting
        early_stopping(avg_er, stFinder)
        if epoch%5==0:
            torch.save(stFinder, 'ST_Finder.pth')

        # Early stopping setting
        if config.using_early_stop and early_stopping.early_stop:
            print("--------------- Early stopping ------------------")
            stFinder.load_state_dict(torch.load('best_network_early_stopping.pth'))
            break

    # test performance
    avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
    for test, valid, time_seq, target_time in testsloader:
        pred = stFinder(test, time_seq, target_time)

        # compute accuracy
        pred, valid = pred / datasets.scaler, valid / datasets.scaler  # scaler is to abilene, the very small value of abilene make pred is all 0
        recov_mat_missing = (test[test.shape[2] // 2 - 1] / datasets.scaler).view(config.batch_size, -1)
        recov_mat = (pred[:, :1]).view(config.batch_size * config.predict_matrix_num, -1)  # recover the input mat at left middle
        recov_valid = (valid[:, :, :1]).view(config.batch_size * config.predict_matrix_num, -1)
        interpo_mat = (pred[:, 1:]).view(config.batch_size * config.predict_matrix_num, -1)  # interpolate matrix
        interpo_valid = (valid[:, :, 1:]).view(config.batch_size * config.predict_matrix_num, -1)

        aer, anmae, armse = get_all_metrices(recov_mat, recov_valid, recov_mat_missing, interpo_mat, interpo_valid)
        avg_er += aer.detach()
        avg_nmae += anmae.detach()
        avg_rmse += armse.detach()
        counter += 1
    print(f'Test_ER=\t{avg_er / counter}\tTest_NMAE=\t{avg_nmae / counter}\tTest_RMSE=\t{avg_rmse / counter}')
    # save model
    torch.save(stFinder, 'ST_Finder.pth')
    return avg_er / counter, avg_nmae / counter, avg_rmse / counter


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def batch_training(config, config2):
    """ batch training in different config """
    sr_list = [[0.8, 0.5], [0.6, 0.5], [0.4, 0.5], [0.2, 0.5], [0.1, 0.5]]
    result_list = []
    for sr1, sr2 in sr_list:
        config.sampling_rate1 = sr1
        config.sampling_rate2 = sr2
        aer, anmae, armse = train_model(config, config2)
        result_list.append([aer.cpu().detach(), anmae.cpu().detach(), armse.cpu().detach()])
        del aer, anmae, armse
    print(f'\nDataset={config2.name} seed={config.seed}\nsr_list:{sr_list}\n:', result_list)
    result_list = np.array(result_list)

    files_path = f'./results/Result_{config2.name}_{time.time()}.txt'
    np.savetxt(files_path, result_list)

    with open(files_path, mode='a') as f:
        f.writelines(f'\n{sr_list}\n')
        f.writelines(f'using_tube_samping={config.using_tube_samping}\n')
        f.writelines(f'dataset={config2.name}\n')
        f.writelines(f'seed={config.seed}\n')


if __name__ == '__main__':
    """ 训练模型 """
    set_seed(config.seed)

    batch_training(config, configAb)  # abilene
    # batch_training(config, configGT)  # GEANT

    # train_model(config, configAb)  # abilene
    # train_model(config, configGT)  # GEANT





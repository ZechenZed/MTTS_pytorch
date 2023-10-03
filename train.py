import os
import json
import torch
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from neural_methods.model.TS_CAN import TSCAN
from data_loader import V4V_Dataset
from tqdm import tqdm
from evaluation.metrics import calculate_metrics
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import MSELoss, L1Loss
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import wandb


class TSCAN_trainer:
    def __init__(self, setup):
        ################### Env setup ###################
        if setup.device_type == 'local':
            self.model_dir = 'C:/Users/Zed/Desktop/MTTS_pytorch/model_ckpts/'
        else:
            self.model_dir = '/edrive1/zechenzh/model_ckpts/'
        self.model_file_name = f'TSCAN_{setup.image_type}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ################ Hyperparameters ################
        self.frame_depth = setup.frame_depth
        self.nb_epoch = setup.nb_epoch
        self.lr = setup.lr
        self.nb_dense = setup.nb_dense
        self.criterion = L1Loss()
        self.min_valid_loss = None
        self.best_epoch = setup.best
        self.base_len = setup.nb_device * self.frame_depth
        self.batch_size = setup.nb_batch
        self.nb_filters1 = setup.nb_filter1
        self.nb_filters2 = setup.nb_filter2
        self.drop_rate1 = setup.dropout_rate1
        self.drop_rate2 = setup.dropout_rate2
        self.kernel = setup.kernel
        self.pool_size = (2, 2)
        self.USE_LAST_EPOCH = True
        self.plot_pred = True

        ################### Load data ###################
        if setup.device_type == 'local':
            data_folder_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
        else:
            data_folder_path = '/edrive1/zechenzh/preprocessed_v4v/'

        self.model = TSCAN(frame_depth=self.frame_depth, img_size=72, dropout_rate1=self.drop_rate1,
                           dropout_rate2=self.drop_rate2, kernel_size=self.kernel, nb_dense=self.nb_dense,
                           pool_size=self.pool_size, nb_filters1=self.nb_filters1,
                           nb_filters2=self.nb_filters2).to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(setup.nb_device)))

        v4v_data_train = V4V_Dataset(data_folder_path, 'train', setup.image_type, setup.BP_type)
        self.train_loader = DataLoader(dataset=v4v_data_train, batch_size=self.batch_size,
                                       shuffle=True, num_workers=1)
        # v4v_data_valid = V4V_Dataset(data_folder_path, 'valid', setup.image_type, setup.BP_type)
        # self.valid_loader = DataLoader(dataset=v4v_data_valid, batch_size=self.batch_size,
        #                                shuffle=True, num_workers=1)
        v4v_data_test = V4V_Dataset(data_folder_path, 'test', setup.image_type, setup.BP_type)
        self.test_loader = DataLoader(dataset=v4v_data_test, batch_size=self.batch_size,
                                      shuffle=False, num_workers=0)

        if self.train_loader and self.test_loader:
            print('Successfully loaded')
        print(f'The Length of train data loder:{len(self.train_loader)}')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr,
                                    epochs=self.nb_epoch, steps_per_epoch=len(self.train_loader))

    def train(self):
        for epoch in range(self.nb_epoch):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(self.train_loader, ncols=80)
            for idx, (data, labels) in enumerate(tbar):
                # print(ind, type(data), type(labels))
                tbar.set_description("Train epoch %s" % epoch)
                data = data.to(self.device)
                labels = labels.to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            # valid_loss = self.valid()
            # print('validation loss: ', valid_loss)
            # if self.min_valid_loss is None:
            #     self.min_valid_loss = valid_loss
            #     self.best_epoch = 0
            #     print("Update best model! Best epoch: {}".format(self.best_epoch))
            # elif valid_loss < self.min_valid_loss:
            #     self.min_valid_loss = valid_loss
            #     self.best_epoch = epoch
            #     print("Update best model! Best epoch: {}".format(self.best_epoch))
            # wandb.log({'train_loss': np.average(train_loss), 'valid_loss': valid_loss})

            wandb.log({'train_loss': np.average(train_loss)})

        # print("")
        # print("Best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self):
        print('')
        print("==========Validating==========")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(self.valid_loader, ncols=80)
            for valid_idx, (val_data, val_labels) in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid = val_data.to(self.device)
                labels_valid = val_labels.to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                # pred_ppg_valid = gaussian_filter(pred_ppg_valid, sigma=25)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def save_model(self, index):
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def test(self):
        print('')
        print("===Testing===")
        if self.USE_LAST_EPOCH:
            last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.nb_epoch - 1) + '.pth')
            print("Testing uses last epoch as non-pretrained model!")
            print(last_epoch_model_path)
            self.model.load_state_dict(torch.load(last_epoch_model_path))
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses best epoch selected using model selection as non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        predictions = list()
        labels = list()
        with torch.no_grad():
            for train_ind, (data_train, train_labels) in enumerate(self.train_loader):
                # batch_size = train_batch[0].shape[0]
                data_train = data_train.to(self.device)
                labels_train = train_labels.to(self.device)
                N, D, C, H, W = data_train.shape
                data_train = data_train.view(N * D, C, H, W)
                labels_train = labels_train.view(-1, 1)

                data_train = data_train[:(N * D) // self.base_len * self.base_len]
                labels_train = labels_train[:(N * D) // self.base_len * self.base_len]

                pred_ppg_train = self.model(data_train)

                pred = pred_ppg_train.detach().cpu().numpy()
                pred = gaussian_filter(pred, sigma=3)
                predictions.append(pred)

                label = labels_train.detach().cpu().numpy()
                labels.append(label)

            predictions = np.array(predictions).reshape(-1)
            labels = np.array(labels).reshape(-1)
            cMAE = sum(abs(predictions - labels)) / predictions.shape[0]
            ro = pearsonr(predictions, labels)[0]
            if np.isnan(ro):
                ro = -1
            wandb.log({'Train_cMAE': cMAE, 'Train_pearson': ro})
            print(f'Train Pearson correlation: {ro}')
            print(f'Train cMAE: {cMAE}')
            if self.plot_pred:
                plt.plot(predictions, 'r', label='Prediction')
                plt.plot(labels, 'g', label='Ground truth')
                plt.legend()
                plt.show()

        # predictions = list()
        # labels = list()
        # with torch.no_grad():
        #     for valid_ind, (data_valid, valid_labels) in enumerate(self.valid_loader):
        #         # batch_size = valid_batch[0].shape[0]
        #         data_valid = data_valid.to(self.device)
        #         labels_valid = valid_labels.to(self.device)
        #         N, D, C, H, W = data_valid.shape
        #         data_valid = data_valid.view(N * D, C, H, W)
        #         labels_valid = labels_valid.view(-1, 1)
        #
        #         data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
        #         labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
        #
        #         pred_ppg_valid = self.model(data_valid)
        #
        #         pred = pred_ppg_valid.detach().cpu().numpy()
        #         pred = gaussian_filter(pred, sigma=3)
        #         predictions.append(pred)
        #
        #         label = labels_valid.detach().cpu().numpy()
        #         labels.append(label)
        #
        #     predictions = np.array(predictions).reshape(-1)
        #     labels = np.array(labels).reshape(-1)
        #     cMAE = sum(abs(predictions - labels)) / predictions.shape[0]
        #     ro = pearsonr(predictions, labels)[0]
        #     wandb.log({'Valid_cMAE': cMAE, 'Valid_pearson': ro})
        #     print(f'Valid Pearson correlation: {ro}')
        #     print(f'Valid cMAE: {cMAE}')
        #     if self.plot_pred:
        #         plt.plot(predictions, 'r', label='Prediction')
        #         plt.plot(labels, 'g', label='Ground truth')
        #         plt.legend()
        #         plt.show()

        predictions = list()
        labels = list()
        with torch.no_grad():
            for test_ind, (data_test, test_labels) in enumerate(self.test_loader):
                # batch_size = test_batch[0].shape[0]
                data_test = data_test.to(self.device)
                labels_test = test_labels.to(self.device)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)

                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]

                pred_ppg_test = self.model(data_test)

                pred = pred_ppg_test.detach().cpu().numpy()
                pred = gaussian_filter(pred, sigma=3)
                predictions.append(pred)

                label = labels_test.detach().cpu().numpy()
                labels.append(label)

            predictions = np.array(predictions).reshape(-1)
            labels = np.array(labels).reshape(-1)
            cMAE = sum(abs(predictions - labels)) / predictions.shape[0]
            ro = pearsonr(predictions, labels)[0]
            if np.isnan(ro):
                ro = -1
            wandb.log({'Test_cMAE': cMAE, 'Test_pearson': ro})
            print(f'Test Pearson correlation: {ro}')
            print(f'Test cMAE: {cMAE}')
            if self.plot_pred:
                plt.plot(predictions, 'r', label='Prediction')
                plt.plot(labels, 'g', label='Ground truth')
                plt.legend()
                plt.show()


if __name__ == '__main__':
    wandb.init(project='TSCAN', config=wandb.config)
    config = wandb.config

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--device_type', type=str, default='remote',
                        help='Local / Remote device')

    parser.add_argument('--nb_device', type=int, default=1,
                        help='Total number of device')

    parser.add_argument('-image', '--image_type', type=str, default='face_large',
                        help='choose from 1) ratio, 2) face_large, 3) face')
    parser.add_argument('-BP', '--BP_type', type=str, default='systolic',
                        help='Choose type of BP from mean, systolic and diastolic')

    parser.add_argument('--nb_epoch', type=int, default=30,
                        help='nb_epoch')
    parser.add_argument('--nb_batch', type=int, default=12,
                        help='nb_batch')
    parser.add_argument('--kernel', type=int, default=3,
                        help='Kernel size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--frame_depth', type=int, default=10,
                        help='frame depth')
    parser.add_argument('--dropout_rate1', type=float, default=0.8128892135118411,
                        help='Drop rate 1')
    parser.add_argument('--dropout_rate2', type=float, default=0.3766033175489906,
                        help='Drop rate 2')
    parser.add_argument('--nb_filter1', type=int, default=16,
                        help='number of filter 1')
    parser.add_argument('--nb_filter2', type=int, default=64,
                        help='number of filter 2')
    parser.add_argument('--nb_dense', type=int, default=2048,
                        help='Number of dense layer')
    parser.add_argument('--best', type=int, default=19,
                        help='Best Epoch')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    trainer = TSCAN_trainer(args)
    trainer.train()
    trainer.test()

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
from torch.nn import MSELoss
import matplotlib.pyplot as plt


class TSCAN_trainer:
    def __init__(self, setup):
        ################### Env setup ###################
        os.environ['CUDA_VISIBLE_DEVICES'] = setup.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_depth = setup.frame_depth
        self.nb_epoch = setup.nb_epoch
        self.lr = setup.lr
        self.criterion = MSELoss()
        self.min_valid_loss = None
        self.best_epoch = 478
        if setup.device_type == 'local':
            self.model_dir = 'C:/Users/Zed/Desktop/MTTS_pytorch/model_ckpts/'
        else:
            self.model_dir = '/edrive2/zechenzh/model_ckpts/'
        self.model_file_name = f'TSCAN_{setup.image_type}'
        self.base_len = setup.nb_device * self.frame_depth
        self.batch_size = setup.nb_batch
        self.USE_LAST_EPOCH = False
        self.plot_pred = True
        self.drop_rate1 = 0.1
        self.drop_rate2 = 0.25
        self.kernel = 12
        ################### Load data ###################
        if setup.device_type == 'local':
            data_folder_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
        else:
            data_folder_path = '/edrive2/zechenzh/preprocessed_v4v/'
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=72, dropout_rate1=self.drop_rate1,
                           dropout_rate2=self.drop_rate2,kernel_size=self.kernel).to(self.device)
        # self.model = torch.nn.DataParallel(model, device_ids=list(range(setup.nb_device)))
        if setup.data_type == 'train':
            print('Loading Data')
            v4v_data_train = V4V_Dataset(data_folder_path, 'train', setup.image_type, setup.BP_type)
            self.train_loader = DataLoader(dataset=v4v_data_train, batch_size=self.batch_size,
                                           shuffle=True, num_workers=1)
            v4v_data_valid = V4V_Dataset(data_folder_path, 'valid', setup.image_type, setup.BP_type)
            self.valid_loader = DataLoader(dataset=v4v_data_valid, batch_size=self.batch_size,
                                           shuffle=True, num_workers=1)
            test = iter(self.valid_loader)
            first_test = next(test)
            index, x, y = first_test
            print(index, type(x), type(y))
            if self.train_loader and self.valid_loader:
                print('Successfully loaded')
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr,
                                        epochs=self.nb_epoch, steps_per_epoch=len(self.train_loader))
        else:
            v4v_data_test = V4V_Dataset(data_folder_path, 'test', setup.image_type, setup.BP_type)
            self.test_loader = DataLoader(dataset=v4v_data_test, batch_size=self.batch_size,
                                          shuffle=True, num_workers=0)
            self.chunk_len = len(self.test_loader)

    def train(self):
        for epoch in range(self.nb_epoch):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(self.train_loader, ncols=80)
            for idx, (_, data, labels) in enumerate(tbar):
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
            valid_loss = self.valid()
            print('validation loss: ', valid_loss)
            if self.min_valid_loss is None:
                self.min_valid_loss = valid_loss
                self.best_epoch = 0
                print("Update best model! Best epoch: {}".format(self.best_epoch))
            elif valid_loss < self.min_valid_loss:
                self.min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
        print("")
        print("Best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self):
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(self.valid_loader, ncols=80)
            for valid_idx, (_, val_data, val_labels) in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid = val_data.to(self.device)
                labels_valid = val_labels.to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
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
        predictions = dict()
        labels = dict()

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
        with torch.no_grad():
            for ind, (_, data_test, test_labels) in enumerate(self.test_loader):
                # batch_size = test_batch[0].shape[0]
                data_test = data_test.to(self.device)
                labels_test = test_labels.to(self.device)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                # for idx in range(batch_size):
                #     subj_index = test_batch[2][idx]
                #     sort_index = int(test_batch[3][idx])
                #     if subj_index not in predictions.keys():
                #         predictions[subj_index] = dict()
                #         labels[subj_index] = dict()
                #     predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:
                #                                                         (idx + 1) * self.chunk_len]
                #     labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
            if self.plot_pred:
                pred = pred_ppg_test.detach().cpu().numpy()
                print(pred)
                label = labels_test.detach().cpu().numpy()
                plt.plot(pred, 'r')
                plt.plot(label, 'g')
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_type', type=str, default='train',
                        help='data type')
    parser.add_argument('-BP', '--BP_type', type=str, default='systolic',
                        help='Choose type of BP from mean, systolic and diastolic')
    parser.add_argument('-image', '--image_type', type=str, default='face_large',
                        help='choose from 1) ratio, 2) face_large, 3) face')
    parser.add_argument('-device', '--device_type', type=str, default='local',
                        help='Local / Remote device')
    parser.add_argument('-g', '--nb_epoch', type=int, default=100,
                        help='nb_epoch')
    parser.add_argument('--nb_batch', type=int, default=6,
                        help='nb_batch')
    parser.add_argument('--gpu', type=str, default='0',
                        help='List of GPUs used')
    parser.add_argument('--nb_device', type=int, default=1,
                        help='Total number of device')
    parser.add_argument('-lr', '--lr', type=float, default=9e-3,
                        help='learning rate')
    parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                        help='frame depth')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args
    trainer = TSCAN_trainer(args)
    if args.data_type == 'train':
        trainer.train()
    else:
        trainer.test()

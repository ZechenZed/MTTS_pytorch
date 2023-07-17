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

parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--exp_type', type=str, default='video',
                    help='experiment type: model or video')
parser.add_argument('-data', '--data_type', type=str, default='train',
                    help='data type')
parser.add_argument('-BP', '--BP_type', type=str, default='systolic',
                    help='Choose type of BP from mean, systolic and diastolic')
parser.add_argument('-image', '--image_type', type=str, default='face_large',
                    help='choose from 1) ratio, 2) face_large, 3) face')
parser.add_argument('-device', '--device_type', type=str, default='remote',
                    help='Local / Remote device')
parser.add_argument('-g', '--nb_epoch', type=int, default=30,
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

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.device_type == 'local':
    data_folder_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
else:
    data_folder_path = '/edrive2/zechenzh/preprocessed_v4v/'

################### Load data ###################
v4v_data = V4V_Dataset(data_folder_path, args.data_type, args.image_type, args.BP_type)
train_loader = DataLoader(dataset=v4v_data, batch_size=4, shuffle=True, num_workers=0)
# dataiter = iter(train_loader)
# data = next(dataiter)
# frame, BP = data
# print(frame, frame.shape)
# print(BP, BP.shape)

################### Model Init ###################
model = TSCAN(frame_depth=args.frame_depth, img_size=72).to(device)
model = torch.nn.DataParallel(model, device_ids=list(range(args.nb_device)))
base_len = args.nb_device * args.frame_depth
criterion = torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.nb_epoch,
                                                steps_per_epoch=len(train_loader))
for epoch in range(args.nb_epoch):
    print('')
    print(f"====Training Epoch: {epoch}====")
    running_loss = 0.0
    train_loss = []
    model.train()
    # Model Training
    tbar = tqdm(train_loader, ncols=80)
    for idx, batch in enumerate(tbar):
        tbar.set_description("Train epoch %s" % epoch)
        data, labels = batch[0].to(
            device), batch[1].to(device)
        N, D, C, H, W = data.shape
        data = data.view(N * D, C, H, W)
        labels = labels.view(-1, 1)
        data = data[:(N * D) // base_len * base_len]
        labels = labels[:(N * D) // base_len * base_len]
        optimizer.zero_grad()
        pred_ppg = model(data)
        loss = criterion(pred_ppg, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if idx % 100 == 99:  # print every 100 mini-batches
            print(
                f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
        train_loss.append(loss.item())
        tbar.set_postfix(loss=loss.item())
#     save_model(epoch)
#     if not config.TEST.USE_LAST_EPOCH:
#         valid_loss = valid(data_loader)
#         print('validation loss: ', valid_loss)
#         if min_valid_loss is None:
#             min_valid_loss = valid_loss
#             best_epoch = epoch
#             print("Update best model! Best epoch: {}".format(best_epoch))
#         elif (valid_loss < min_valid_loss):
#             min_valid_loss = valid_loss
#             best_epoch = epoch
#             print("Update best model! Best epoch: {}".format(best_epoch))
# if not config.TEST.USE_LAST_EPOCH:
#     print("best trained epoch: {}, min_val_loss: {}".format(best_epoch, min_valid_loss))


import time
import os
import logging
from tqdm import tqdm

from utils import unet_dataset
from models import unetFEGcn
from metrics import eval_metrics
#from predict import predict
from lr_schedule import step_lr, exp_lr_scheduler


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(config):

    # train配置
    device = torch.device('cuda:0')
    selected = config['train_model']['model'][config['train_model']['select']]
    if selected  == 'unetFEGcn':
        model = unetFEGcn.UNet(num_classes=config['num_classes'])

    model.to(device)

    logger = initLogger(selected)

    # loss
    criterion = nn.CrossEntropyLoss()

    # train data
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.209,0.394,0.380,0.344,0.481],std=[0.141,0.027,0.032,0.046,0.069])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )
    dst_train = unet_dataset.UnetDataset(config['train_list'], transform=transform,train=True)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config['batch_size'])

    # validation data
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.209,0.394,0.380,0.344,0.481],std=[0.141,0.027,0.032,0.046,0.069])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )
    dst_valid = unet_dataset.UnetDataset(config['test_list'], transform=transform,train=False)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config['batch_size'])

    cur_acc = []
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'], 0.999], weight_decay=config['weight_decay'])
    #最优val准确率，根据这个保存模型
    val_max_pixACC = 0.0
    val_min_loss = 100.0
    for epoch in range(config['num_epoch']):
        epoch_start = time.time()
        # lr
        
        
        
        model.train()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        IoU = 0.0
        tbar = tqdm(dataloader_train, ncols=120)

        #混淆矩阵
        conf_matrix_train = np.zeros((config['num_classes'],config['num_classes']))

        for batch_idx, (data, target,path) in enumerate(tbar):
            tic = time.time()

            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(output, target, config['num_classes'],conf_matrix_train)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1)+labeled_sum)
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            tbar.set_description('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config['batch_size']),
                pixelAcc, IoU.mean(),
                time.time()-tic, time.time()-epoch_start))
            cur_acc.append(pixelAcc)

        logger.info('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {}  mIoU {:.5f} '.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(IoU), IoU.mean()))
            

        # val
        test_start = time.time()
        
        model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        mIoU = 0.0
        tbar = tqdm(dataloader_valid, ncols=120)
        class_precision=np.zeros(config['num_classes'])
        class_recall=np.zeros(config['num_classes'])
        class_f1=np.zeros(config['num_classes'])
        # val_list=[]
        
                # data, target = data.to(device), target.to(device)
        with torch.no_grad():
            #混淆矩阵
            conf_matrix_val = np.zeros((config['num_classes'],config['num_classes']))
            for batch_idx, (data, target,path) in enumerate(tbar):
                tic = time.time()
                
                output = model(data)
                loss = criterion(output, target)
                loss_sum += loss.item()

                correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(output, target, config['num_classes'], conf_matrix_val)
                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                for i in range(config['num_classes']):
                    #每一类的precision
                    class_precision[i]=1.0*conf_matrix_val[i,i]/conf_matrix_val[:,i].sum()
                    #每一类的recall
                    class_recall[i]=1.0*conf_matrix_val[i,i]/conf_matrix_val[i].sum()
                    #每一类的f1
                    class_f1[i]=(2.0*class_precision[i]*class_recall[i])/(class_precision[i]+class_recall[i])

                tbar.set_description('VAL ({}) | Loss: {:.5f} | Acc {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                    epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                    pixelAcc, mIoU.mean(),
                            time.time() - tic, time.time() - test_start))
            if loss_sum < val_min_loss:
                val_min_loss = loss_sum
                best_epoch =np.zeros(2)
                best_epoch[0]=epoch
                best_epoch[1]=conf_matrix_val.sum()
                if os.path.exists(config['save_model']['save_path']) is False:
                    os.mkdir(config['save_model']['save_path'])
                torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], selected+'_jx.pth'))
                np.savetxt(os.path.join(config['save_model']['save_path'],  selected+'_conf_matrix_val.txt'),conf_matrix_val,fmt="%d")
                np.savetxt(os.path.join(config['save_model']['save_path'], selected+'_best_epoch.txt'),best_epoch)
        logger.info('VAL ({}) | Loss: {:.5f} | OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(mIoU), mIoU.mean(),toString(class_precision),toString(class_recall),toString(class_f1)))

def toString(IOU):
    result = '{'
    for i, num in enumerate(IOU):
        result += str(i) + ': ' + '{:.4f}, '.format(num)

    result += '}'
    return result

def initLogger(model_name):
    # 初始化log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = r'logs'
    log_name = os.path.join(log_path, "new"+model_name + '_jx_new_metrics' + rq + '.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
if __name__ == '__main__':
    # train()
    while True:
        print(1)
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import AGINDataset
from .utils.function import performance_fit
# from .utils.function import Fidelity_Loss
import models.JOINT as JOINT
import random
import pandas as pd
# from .utils.get_idx import get_idx
# from .utils.dataaug import get_transforms_
from .utils.function import Monotonicity_Loss

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="AIGI Naturalness Assessment")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=80, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=32, type=int)
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--lr_weight_L2', type=float, default=1)
    parser.add_argument('--lr_weight_pair', type=float, default=1)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--snapshot', default='./savedir/', type=str)

    parser.add_argument('--pretrained_path_T', type=str, default='')
    parser.add_argument('--pretrained_path_R', type=str, default='./models/Model_SwinT_AVA_epoch_10.pth')
    
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--database_dir', type=str, default='/home/czj/dataset/AGIN_512x512')   # AGIN_partitioned
    parser.add_argument('--model', default='JOINT', type=str)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--print_samples', type=int, default=50)
    parser.add_argument('--database', default='AGIN', type=str)
    parser.add_argument('--crop_size', type=int,default=224)

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    gpu = args.gpu
    cudnn.enabled = True
    crop_size = args.crop_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    decay_interval = args.decay_interval
    decay_ratio = args.decay_ratio
    snapshot = args.snapshot
    database = args.database
    print_samples = args.print_samples
    results_path = args.results_path
    database_dir = args.database_dir
    resize = args.resize

    seed = args.random_seed
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename_list = './AGIN/MOS_AGIN.csv'


    model = JOINT.JOINT_Model(pretrained_path_T=args.pretrained_path_T, pretrained_path_R=None)

    transforms_train = transforms.Compose([transforms.Resize(resize),
                                           transforms.RandomCrop(crop_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    transforms_test = transforms.Compose([transforms.Resize(resize),
                                          transforms.CenterCrop(crop_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    train_dataset = AGINDataset.AGIN_dataloader(database_dir,
                                                    filename_list,
                                                    transforms_train,
                                                    database + '_train',
                                                    seed)
    test_dataset = AGINDataset.AGIN_dataloader(database_dir,
                                              filename_list,
                                              transforms_test,
                                              database + '_test',
                                              seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=8)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    criterion = Monotonicity_Loss().to(device)
    criterion2 = nn.MSELoss().to(device)




    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=0.0000002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=decay_interval,
                                                gamma=decay_ratio)
    print("Ready to train network")

    best_test_criterion = -1  # SROCC min
    best = np.zeros(5)

    n_train = len(train_dataset)
    n_test = len(test_dataset)


    for epoch in range(num_epochs):
        # train
        model.train()

        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, data_train in enumerate(train_loader):
            img_T = data_train['img_technical'].to(device)
            img_R = data_train['img_rationality'].to(device)
            
            
            mos = data_train['y_label'][:, np.newaxis]
            mos_T = data_train['y_label_tech'][:, np.newaxis]
            mos_R = data_train['y_label_rat'][:, np.newaxis]
            
            mos = mos.to(device)
            mos_T = mos_T.to(device)
            mos_R = mos_R.to(device)
            
            mos_output_T, mos_output_R = model(img_T, img_R)


            optimizer.zero_grad()

            # base
            loss = criterion(mos_output_T, mos_T)+criterion2(mos_output_T, mos_T)+criterion2(mos_output_R, mos_R)+criterion(mos_output_R, mos_R)

            
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % print_samples == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                print('Epoch: {:d}/{:d} | Step: {:d}/{:d} | Training loss: {:.4f}'.format(epoch + 1,
                                                                                          num_epochs,
                                                                                          i + 1,
                                                                                          len(train_dataset) // batch_size,
                                                                                          avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
        # print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
        print('Epoch {:d} averaged training loss: {:.4f}'.format(epoch + 1, avg_loss))

        scheduler.step()
        lr_current = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr_current[0]))

        # Test
        model.eval()
        y_output = np.zeros(n_test)
        y_output_T = np.zeros(n_test)
        y_output_R = np.zeros(n_test)
        y_test = np.zeros(n_test)
        y_test_T = np.zeros(n_test)
        y_test_R = np.zeros(n_test)

        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                img_T = data_test['img_technical'].to(device)
                img_R = data_test['img_rationality'].to(device)

                y_test[i] = data_test['y_label'].item()
                y_test_T[i] = data_test['y_label_tech'].item()
                y_test_R[i] = data_test['y_label_rat'].item()
                
                
                mos = mos.to(device)
                mos_T = mos_T.to(device)
                mos_R = mos_R.to(device)
                
                outputs_T, outputs_R = model(img_T, img_R)

                y_output[i] = outputs_T.item()+outputs_R.item()
                
                
            test_PLCC, test_SRCC, test_KRCC, test_RMSE, test_MAE,popt = performance_fit(y_test, y_output)
           
            
            print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE))
           
            if test_SRCC + test_PLCC + test_KRCC - test_RMSE - test_MAE > best_test_criterion:
                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)
                    if os.path.exists(old_save_name_popt):
                        os.remove(old_save_name_popt)

                save_model_name = os.path.join(args.snapshot,
                                               args.model + '_' + args.database + '_' + '_NR_' + 'Swin_Res_lossTR_epoch_%d_SRCC_%f.pth' % (
                                               epoch + 1, test_SRCC))
                save_popt_name = os.path.join(args.snapshot,
                                              args.model + '_' + args.database + '_' + '_NR_' + 'Swin_Res_lossTR_epoch_%d_SRCC_%f.npy' % (
                                              epoch + 1, test_SRCC))
                print("Update best model using best_val_criterion ")
                torch.save(model.state_dict(), save_model_name)
                np.save(save_popt_name, popt)
                old_save_name = save_model_name
                old_save_name_popt = save_popt_name
                best[0:5] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE]
                best_popt = popt
                best_test_criterion = test_SRCC + test_PLCC + test_KRCC - test_RMSE - test_MAE 

                print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(
                    test_SRCC, test_KRCC, test_PLCC, test_RMSE, test_MAE))


        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, MAE={:.4f}".format(best[0],
                                                                                                              best[1],
                                                                                                              best[2],
                                                                                                              best[3],
                                                                                                              best[4]))
        print(
            '*************************************************************************************************************************')
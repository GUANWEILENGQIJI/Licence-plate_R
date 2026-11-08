# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

"""
--max_epoch       训练轮数，默认 15
--train_img_dirs  训练图片目录
--test_img_dirs   测试图片目录
--train_batch_size 训练批次大小，默认 128
--test_batch_size  测试批次大小，默认 120
--learning_rate   学习率，默认 0.1
--num_workers     数据加载线程数
--save_folder     模型保存目录
--pretrained_model 使用已有 .pth 权重继续训练
--resume_epoch    从指定轮次继续

python .\train_LPRNet.py `
  --train_img_dirs ".\data\wrong_train" `
  --test_img_dirs ".\data\test2" `
  --save_folder ".\weights" `
  --max_epoch 200 `
  --train_batch_size 32 `
  --test_batch_size 128 `
  --learning_rate 0.0001 `
  --lr_schedule 120 200 260 `
  --num_workers 0 `
  --pretrained_model ".\weights\run_\Final_LPRNet_model.pth"
"""

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = base_lr * (0.1 ** len(lr_schedule))
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="~/workspace/trainMixLPR", help='the train images path')
    parser.add_argument('--test_img_dirs', default="~/workspace/testMixLPR", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, type=int, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument(
        '--lr_schedule',
        default=[40, 70, 90],
        nargs='+',
        type=int,
        help='epochs at which to reduce the learning rate.',
    )
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def create_run_folder(base_folder):
    os.makedirs(base_folder, exist_ok=True)
    run_number = 1
    while True:
        run_folder = os.path.join(base_folder, f"run_{run_number}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        run_number += 1

def save_training_curves(save_folder, epochs, loss_history, accuracy_history):
    """Save the training loss and test accuracy curves and their raw history."""
    if not epochs:
        return

    history_path = os.path.join(save_folder, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as history_file:
        json.dump(
            {
                'epoch': epochs,
                'loss': loss_history,
                'accuracy': accuracy_history,
            },
            history_file,
            ensure_ascii=False,
            indent=2,
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, loss_history, marker='o', color='tab:blue', label='Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CTC Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    accuracy_epochs = epochs[:len(accuracy_history)]
    axes[1].plot(accuracy_epochs, accuracy_history, marker='o', color='tab:green', label='Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    curve_path = os.path.join(save_folder, 'training_curves.png')
    figure.savefig(curve_path, dpi=200)
    plt.close(figure)

def train():
    args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0
    loss_batch_count = 0
    epoch_history = []
    loss_history = []
    accuracy_history = []

    save_root = os.path.abspath(args.save_folder)
    args.save_folder = create_run_folder(save_root)
    print(f"Model output directory: {args.save_folder}")

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    if len(train_dataset) == 0:
        raise ValueError('The training dataset is empty. Please check --train_img_dirs.')
    if len(test_dataset) == 0:
        raise ValueError('The test dataset is empty. Please check --test_img_dirs.')
    epoch_size = (len(train_dataset) + args.train_batch_size - 1) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            loss_batch_count = 0
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_folder,
                'LPRNet_' + '_iteration_' + repr(iteration) + '.pth',
            )
            torch.save(lprnet.state_dict(), checkpoint_path)

        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(lprnet, test_dataset, args)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss_item = loss.item()
        if np.isfinite(loss_item):
            loss.backward()
            optimizer.step()
            loss_val += loss_item
            loss_batch_count += 1
        else:
            print('[Warning] Non-finite loss at iteration {}, skipping optimizer update.'.format(iteration))
        end_time = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss_item) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))

        if (iteration + 1) % epoch_size == 0:
            average_loss = loss_val / max(loss_batch_count, 1)
            epoch_history.append(epoch)
            loss_history.append(average_loss)
            epoch_accuracy = Greedy_Decode_Eval(lprnet, test_dataset, args)
            accuracy_history.append(epoch_accuracy)
            save_training_curves(args.save_folder, epoch_history, loss_history, accuracy_history)
            print('[Info] Epoch {} average loss: {:.6f}'.format(epoch, average_loss))
    # final test
    print("Final test Accuracy:")
    final_accuracy = Greedy_Decode_Eval(lprnet, test_dataset, args)

    # Keep the final curve point in sync with the final evaluation.
    if accuracy_history:
        accuracy_history[-1] = final_accuracy
    elif epoch_history:
        accuracy_history.append(final_accuracy)

    # save final parameters
    final_model_path = os.path.join(args.save_folder, 'Final_LPRNet_model.pth')
    torch.save(lprnet.state_dict(), final_model_path)
    save_training_curves(args.save_folder, epoch_history, loss_history, accuracy_history)
    print('Training curves saved to: {}'.format(os.path.join(args.save_folder, 'training_curves.png')))
    print('Training history saved to: {}'.format(os.path.join(args.save_folder, 'training_history.json')))

def Greedy_Decode_Eval(Net, datasets, args):
    was_training = Net.training
    Net.eval()
    epoch_size = (len(datasets) + args.test_batch_size - 1) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    with torch.no_grad():
        for i in range(epoch_size):
            # load train data
            images, labels, lengths = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start+length]
                targets.append(label)
                start += length
            targets = [el.numpy() for el in targets]

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = Net(images)
            # greedy decode
            prebs = prebs.cpu().numpy()
            preb_labels = list()
            for i in range(prebs.shape[0]):
                preb = prebs[i, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label: # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            for i, label in enumerate(preb_labels):
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1

    if was_training:
        Net.train()

    total = Tp + Tn_1 + Tn_2
    Acc = Tp * 1.0 / total if total else 0.0
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    return Acc


if __name__ == "__main__":
    train()

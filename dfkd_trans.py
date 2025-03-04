import argparse
from math import gamma
import os
import random
import shutil
import time
import warnings
import pdb
import copy
import sys
import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils_fl import *
from utils_moe import *

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True)
parser.add_argument('--adv', default=1.0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--ohg', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--div', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')

parser.add_argument('--save_dir', default='run/synthesis', type=str)
parser.add_argument('--batchonly', action='store_true')
parser.add_argument('--batchused', action='store_true')
parser.add_argument('--sam', default=0.0, type=float)
parser.add_argument('--his', action='store_false')
parser.add_argument('--wdc', default=0.99, type=float)

################  para to adjust W for ensemble
parser.add_argument('--mv', default=1.0, type=float)
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--mu', default=0.01, type=float)
parser.add_argument('--wa_steps', default=1, type=int)

# Basic
parser.add_argument('--data_root', default='/home/dairong/Co-Boosting-main/Data')
parser.add_argument('--fl_model', default='')
parser.add_argument('--teacher', default='resnet18')
parser.add_argument('--student', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--kd_lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float,
                    help='initial learning rate for generation')

parser.add_argument('--kd_T', default=4, type=float)
parser.add_argument('--odseta', default=8, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')

# co-boosting inherent
parser.add_argument('--ods', action='store_true',
                    help='是否在KD阶段使用ODS技术')
parser.add_argument('--hast', action='store_true',
                    help='是否使用modified CE loss')
parser.add_argument('--hs', default=1.0, type=float, metavar='N',
                    help='number of total iterations in each epoch')
###
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--identity', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--imgsize', default=32, type=int,
                    help='sam')

best_acc1 = 0


def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1_dict, multi_stu_lst, best_acc1
    multi_stu_lst = ['cnn', 'resnet18', 'vgg11', 'mobile', 'shuffle']
    best_acc1_dict = {name: 0 for name in multi_stu_lst}

    ############################################
    # GPU and FP16
    ############################################
    args.autocast = datafree.utils.dummy_ctx

    ############################################
    # Logger
    ############################################
    # pdb.set_trace()
    args.his = not args.batchonly
    log_name = '%s_%s_adv%s_ohg%s_KDlr%s_KDT%s_GANlr%s_GANs%s_Epoch%s_seed%s' % (
        args.method, args.student, args.adv, args.ohg, args.kd_lr, args.kd_T, args.lr_g, args.g_steps, args.epochs,
        args.seed)
    if args.method in ['co_boosting', 'trans']:
        args.weighted = True
        args.hast = True
        args.ods = True
        log_name += '_eta' + str(args.odseta) 
        log_name += '_hast' + str(args.hs)
        args.odseta = args.odseta / 255
        log_name += '_wmu' + str(args.mu) + '_was' + str(args.wa_steps) + '_wdc' + str(args.wdc)

    prefix_path = '/home/dairong/Co-Boosting-main/'

    args.identity = log_name
    args.logger = datafree.utils.logger.get_logger(log_name, output=prefix_path + 'LOG/%s/%s.txt' % (
    args.fl_model, args.identity))
    os.makedirs(prefix_path + 'checkpoints/%s/' % (args.fl_model), exist_ok=True)
    for k, v in datafree.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    ############################################
    # Setup dataset
    ############################################
    num_classes = None; ori_dataset = None; val_dataset = None; val_loader = None; evaluator = None; method_transform = None
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    method_transform = ori_dataset.transform
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    if args.dataset in ["mnist",'fmnist']:
        real_img_size = (1, 32, 32); nc = 1
    elif args.dataset in ["cifar10", 'cifar100']:
        real_img_size = (3, 32, 32); nc = 3
    elif args.dataset in ['tiny']:
        real_img_size = (3, 32, 32); nc = 3

    ############################################
    # Setup models
    ############################################
    student_dict = {}
    if args.student != 'multi':
        single_student = registry.get_model(args.student, num_classes=num_classes)
        single_student = single_student.cuda()
    else:
        for stu_name in multi_stu_lst:
            student_dict[stu_name] = registry.get_model(stu_name, num_classes=num_classes)
        student_dict = {name: stu.cuda() for name, stu in student_dict.items()}

    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    local_weights = torch.load(prefix_path + 'checkpoints/FL_pretrain/%s.pkl' % (args.fl_model))

    model_list = []
    for i in range(len(local_weights)):
        tmp_mdl = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
        net = copy.deepcopy(tmp_mdl)
        net = net.cuda()
        net.load_state_dict(local_weights[i])
        net.eval()
        model_list.append(net)
    ensemble_model = Ensemble(model_list)

    net = MoE(input_size = math.prod(real_img_size), experts = model_list, noisy_gating=True, k=len(model_list))
    ensemble_model = net.cuda()

    moe_optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    
    if args.method == 'co_boosting':
        ww = torch.zeros(size=(len(model_list), 1))
        for _ww in range(len(model_list)):
            ww[_ww] = 1.0 / len(model_list)
        ww = ww.cuda()
        ensemble_model = WEnsemble(model_list, ww)

    teacher = ensemble_model.cuda()
    args.logger.info("NOW TESTING TEACHER MODEL")
    class_test(teacher, val_loader, args.logger)

    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size


    args.save_dir = prefix_path + 'checkpoints/%s/%s/' % (args.fl_model, args.identity)
    # 检查目录是否存在
    if os.path.exists(args.save_dir):
        print("Already Exist, Try another hyperparameter!")
        sys.exit(0)  # 退出程序
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.method ==  'dense':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=real_img_size[-1], nc=nc)
        generator = generator.cuda()
        criterion = datafree.criterions.KLDiv(T=1)
        synthesizer = datafree.synthesis.DENSESynthesizer(
            teacher=teacher, mdl_list=model_list, student=single_student, generator=generator, nz=nz, num_classes=num_classes,
            img_size=real_img_size, iterations=args.g_steps, lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh = args.ohg, criterion=criterion,
            transform=method_transform,
            save_dir=args.save_dir, normalizer=args.normalizer, args=args)

    elif args.method == 'co_boosting':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=real_img_size[-1], nc=nc)
        generator = generator.cuda()
        criterion = datafree.criterions.KLDiv(T=1)
        synthesizer = datafree.synthesis.COBOOSTSynthesizer(
            teacher=teacher, mdl_list=model_list, student=single_student, generator=generator, nz=nz, num_classes=num_classes,
            img_size=real_img_size, iterations=args.g_steps, lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh = args.ohg, criterion=criterion, moe_opt = moe_optimizer,
            transform=method_transform,
            save_dir=args.save_dir, normalizer=args.normalizer, args=args)

    elif args.method == 'trans':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=real_img_size[-1], nc=nc)
        generator = generator.cuda()
        criterion = datafree.criterions.KLDiv(T=1)
        synthesizer = datafree.synthesis.TRANSSynthesizer(
            teacher=teacher, mdl_list=model_list, student_dict=student_dict, generator=generator, nz=nz, num_classes=num_classes,
            img_size=real_img_size, iterations=args.g_steps, lr_g=args.lr_g, moe_opt=moe_optimizer,
            synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh = args.ohg, div = args.div, criterion=criterion,
            transform=method_transform,
            save_dir=args.save_dir, normalizer=args.normalizer, args=args)

    ############################################
    # Setup KD LR optimizer
    ############################################
    if args.student == "multi":
        optimizers = {name: torch.optim.SGD(stu.parameters(), args.kd_lr, weight_decay=args.weight_decay, momentum=0.9) 
                    for name, stu in student_dict.items()}
        schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs) 
                    for name, optim in optimizers.items()}
    else:
        single_optimizer = torch.optim.SGD(single_student.parameters(), args.kd_lr, weight_decay=args.weight_decay, momentum=0.9)
        single_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(single_optimizer, T_max=args.epochs)
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        # for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
        # 1. Data synthesis
        vis_results = synthesizer.synthesize(cur_ep=epoch)  # g_steps
        # 2. Knowledge distillation
        del teacher
        teacher = synthesizer.teacher
        teacher = teacher.cuda()
        kd_criterion = datafree.criterions.KLDiv(T=args.kd_T)
        if args.method ==  'dense':
            dense_kd_train(synthesizer, [single_student, teacher], kd_criterion, single_optimizer, args)  # # kd_steps
        elif args.method ==  'co_boosting':
            cb_kd_train(synthesizer, [single_student, teacher], kd_criterion, single_optimizer, args)  # # kd_steps
        elif args.method ==  'trans':
            trans_kd_train(synthesizer, [student_dict, teacher], kd_criterion, optimizers, args)  # # kd_steps

        if args.student == "multi":
            for name, student in student_dict.items():
                student.eval()
                eval_results = evaluator(student)
                (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
                args.logger.info('[Eval] Model={model_name} Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                                .format(model_name=name, current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss,
                                        lr=optimizers[name].param_groups[0]['lr']))
                schedulers[name].step()

                is_best = acc1 > best_acc1_dict[name]
                best_acc1_dict[name] = max(acc1, best_acc1_dict[name])
                _best_ckpt = prefix_path + 'checkpoints/%s/%s_%s.pth' % (args.fl_model, args.identity, name)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': name,  # 模型名称
                    'state_dict': student.state_dict(),
                    'best_acc1': float(best_acc1_dict[name]),
                    'optimizer': optimizers[name].state_dict(),
                    'scheduler': schedulers[name].state_dict(),
                }, is_best, filename=_best_ckpt)

            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                for name, student in student_dict.items():
                    args.logger.info(f"Now testing {name}")
                    class_test(student, val_loader, args.logger)
                # args.logger.info(teacher.mdl_w_list)
                args.logger.info("Now testing MOE model")
                class_test(teacher, val_loader, args.logger)
        else:
            single_student.eval()
            eval_results = evaluator(single_student)
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
            args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                            .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss,
                                    lr=single_optimizer.param_groups[0]['lr']))
            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                class_test(single_student, val_loader, args.logger)
                args.logger.info(teacher.mdl_w_list)
                args.logger.info("Now testing weighted ENSEMBLE")
                class_test(teacher, val_loader, args.logger)
            single_scheduler.step()
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            _best_ckpt = prefix_path + 'checkpoints/%s/%s_%s.pth' % (args.fl_model, args.identity, args.student)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': single_student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer': single_optimizer.state_dict(),
                'scheduler': single_scheduler.state_dict(),
            }, is_best, filename=_best_ckpt)
    
    if args.student == 'multi':
        for name, best_acc in best_acc1_dict.items():
            args.logger.info("Best accuracy for model {}: {:.4f}".format(name, best_acc))
    else:
        args.logger.info("Best: %.4f" % best_acc1)



def cb_kd_train(synthesizer, model, criterion, optimizer, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(T = args.kd_T, reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    student.train()
    teacher.eval()
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.zero_grad()
        images = images.cuda(); labels = labels.cuda()
        loss_ce = torch.tensor(0).cuda()
        images.requires_grad = True
        try:
            random_w = torch.FloatTensor(*teacher(images, labels).shape).uniform_(-1., 1.).to('cuda')
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images, labels) / 4)).sum()
        except:
            random_w = torch.FloatTensor(*teacher(images).shape).uniform_(-1., 1.).to('cuda')
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images) / 4)).sum()
        loss_ods.backward()
        images = (torch.sign(images.grad) * args.odseta + images).detach()

        s_out = student(images.detach())
        with torch.no_grad():
            try:
                t_out, t_feat = teacher(images, labels, return_features=True)
            except:
                t_out, t_feat = teacher(images, return_features=True)
            try:
                loss_ce = torch.nn.functional.cross_entropy(s_out, labels)
            except:
                continue
        loss_kd = criterion(s_out, t_out.detach())
        loss = loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=10)
        optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and idx % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[KD_Train] Epoch={current_epoch} Iter={i}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, kd_Loss={kd_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=idx, train_acc1=train_acc1, train_acc5=train_acc5,kd_loss=loss_kd.item(), lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()

def trans_kd_train(synthesizer, model, criterion, optimizers, args):
    student_dict, teacher = model

    # 初始化度量器
    loss_metrics = {name: datafree.metrics.RunningLoss(datafree.criterions.KLDiv(T=args.kd_T, reduction='sum')) for name in student_dict}
    acc_metrics = {name: datafree.metrics.TopkAccuracy(topk=(1, 5)) for name in student_dict}

    # 将所有的学生模型设为训练模式
    for name, student in student_dict.items():
        student.train()

    teacher.eval()

    # 遍历训练数据
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        images = images.cuda(); labels = labels.cuda()

        # 遍历每个学生模型
        for name, student in student_dict.items():
            optimizer = optimizers[name]  # 获取对应的优化器
            optimizer.zero_grad()

            # 初始化损失
            loss_ce = torch.tensor(0).cuda()
            images.requires_grad = True

            # 获取教师模型的输出
            random_w = torch.FloatTensor(*teacher(images)[0].shape).uniform_(-1., 1.).to('cuda')
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images)[0] / 4)).sum()

            loss_ods.backward()
            images = (torch.sign(images.grad) * args.odseta + images).detach()

            # 学生模型的输出
            s_out = student(images.detach())

            with torch.no_grad():
                t_out = teacher(images)[0]

            # 计算知识蒸馏损失
            loss_kd = criterion(s_out, t_out.detach())
            loss = loss_kd
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=10)

            # 优化器步进
            optimizer.step()

            # 更新每个学生模型的度量指标
            acc_metrics[name].update(s_out, t_out.max(1)[1])
            loss_metrics[name].update(s_out, t_out)

        # 打印每个学生模型的训练进度
        if args.print_freq > 0 and idx % args.print_freq == 0:
            for name in student_dict:
                (train_acc1, train_acc5), train_loss = acc_metrics[name].get_results(), loss_metrics[name].get_results()
                args.logger.info(
                    '[KD_Train] Model={model_name} Epoch={current_epoch} Iter={i}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, '
                    'kd_Loss={kd_loss:.4f}, Lr={lr:.4f}'.format(
                        model_name=name, current_epoch=args.current_epoch, i=idx, train_acc1=train_acc1, train_acc5=train_acc5,
                        kd_loss=train_loss, lr=optimizers[name].param_groups[0]['lr']
                    )
                )
                loss_metrics[name].reset()
                acc_metrics[name].reset()


def dense_kd_train(synthesizer, model, criterion, optimizer, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(T = args.kd_T, reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    student.train()
    teacher.eval()
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.zero_grad()
        images = images.cuda(); labels = labels.cuda()
        loss_ce = torch.tensor(0).cuda()
        s_out = student(images.detach())
        with torch.no_grad():
            try:
                t_out, t_feat = teacher(images, labels, return_features=True)
            except:
                t_out, t_feat = teacher(images, return_features=True)
            try:
                loss_ce = torch.nn.functional.cross_entropy(s_out, labels)
            except:
                continue
        loss_kd = criterion(s_out, t_out.detach())
        loss = loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=10)
        optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and idx % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[KD_Train] Epoch={current_epoch} Iter={i}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, kd_Loss={kd_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=idx, train_acc1=train_acc1, train_acc5=train_acc5,kd_loss=loss_kd.item(), lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()

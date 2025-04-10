import torch, os, time, model, utils, sys
import numpy as np
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR
from utils import Logger, AverageMeter, ListAverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
from tqdm import tqdm

from utils import hsic
import utils.utils as utils

device = "cuda"


def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        out_digit = model(img)[-1]
        out_iden = torch.argmax(out_digit, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def multilayer_hsic(args, model, criterion, inputs, target, a1, a2, n_classes):
    dxz_list = []
    dyz_list = []
    bs = inputs.size(0)
    total_loss = 0

    bido_dxz_list = []
    bido_dyz_list = []
    if args.hsic_training:
        if args.dataset == 'chestxray':
            hiddens, out_digit = model(inputs, release=False)
        else:
            hiddens, out_digit = model(inputs)

        cross_loss = criterion(out_digit, target)

        total_loss += cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)

        for hidden in hiddens:
            hidden = hidden.view(bs, -1)

            if args.defense == 'HSIC':
                hxz_l, hyz_l = hsic.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=None,
                    ktype=args.ktype
                )

            elif args.defense == 'COCO':
                hxz_l, hyz_l = hsic.coco_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=None,
                    ktype=args.ktype
                )

            dxz_list.append(hxz_l.item())
            dyz_list.append(hyz_l.item())

            bido_dxz_list.append(hxz_l)
            bido_dyz_list.append(hyz_l)

        if args.bido_layer:
            bido_dxz_list, bido_dyz_list = [bido_dxz_list[args.bido_layer - 1]], [bido_dyz_list[args.bido_layer - 1]]

        for (hxz_l, hyz_l) in zip(bido_dxz_list, bido_dyz_list):
            temp_hsic = a1 * hxz_l - a2 * hyz_l
            total_loss += temp_hsic

    return total_loss, cross_loss, out_digit, dxz_list, dyz_list


def train_HSIC(args, model, criterion, optimizer, trainloader, epoch, a1, a2, n_classes):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    dxz_l, dyz_l = ListAverageMeter(args.n_hiddens), ListAverageMeter(args.n_hiddens)

    # pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=140)
    for batch_idx, (inputs, iden) in enumerate(trainloader):
        data_time.update(time.time() - end)
        bs = inputs.size(0)

        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        loss, cross_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(args, model, criterion, inputs,
                                                                          iden, a1, a2, n_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(dxz_list) / len(dxz_list))
        lyz.update(sum(dyz_list) / len(dyz_list))

        dxz_l.update(dxz_list)
        dyz_l.update(dyz_list)

        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
        #       'top1:{top1: .4f} | top5:{top5: .4f}'.format(
        #     cls=loss_cls.avg,
        #     lxz=lxz.avg,
        #     lyz=lyz.avg,
        #     loss=losses.avg,
        #     top1=top1.avg,
        #     top5=top5.avg,
        # )
        # pbar.set_description(msg)
        if batch_idx % args.print_freq == 0:
            # plot progress
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'CE {cls.val:.4f} ({cls.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
                epoch+1, batch_idx, len(trainloader), batch_time=batch_time,
                cls=loss_cls,
                loss=losses,
                top1=top1,
            ))
            # print('(***Lxz(down) {lxz.val:.4f} ({lxz.avg:.4f})\t'
            #         'Lyz(up) {lyz.val:.4f} ({lyz.avg:.4f})***)\n'.format(
            #             lxz=lxz, lyz=lyz))

    return dxz_l.avg, dyz_l.avg, lxz.avg, lyz.avg


def train_HSIC_OOD(args, model, criterion, optimizer, trainloader, publicloader, a1, a2, b, n_classes):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, loss_cls = AverageMeter(), AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    ent = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)
    Entropy = utils.HLoss()

    for batch_idx, (inputs, iden, _) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)

        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        ######################################## clf + BiDO ##########################################
        loss, cross_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(args, model, criterion, inputs,
                                                                          iden, a1, a2, n_classes)

        ######################################## clf + BiDO ##########################################

        ################# OOD ent #################
        OOD_data = publicloader.next()
        OOD_data = OOD_data.to(device)
        _, ood_digit = model(OOD_data)
        entropy = Entropy(ood_digit)
        ################# OOD ent #################
        loss = loss - b * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        top1.update(prec1.item())
        top5.update(prec5.item())

        # record loss
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(dxz_list) / len(dxz_list), bs)
        lyz.update(sum(dyz_list) / len(dyz_list), bs)

        ent.update(entropy.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | ent:{ent:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            ent=ent.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return loss_cls.avg, top1.avg, lxz.avg, lyz.avg, ent.avg


def train_attr(args, model, criterion, optimizer, trainloader, a1, a2, n_classes, epoch, attr_num=20):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=140)
    for batch_idx, (inputs, _, attr) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)

        ##################
        iden = attr[attr_num]
        ##################

        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        loss, cross_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(args, model, criterion, inputs,
                                                                          iden, a1, a2, n_classes)

        if epoch < 15:
            optimizer.zero_grad()
            cross_loss.backward()
            optimizer.step()

        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        prec1, _ = accuracy(out_digit.data, iden.data, topk=(1, 1))
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(dxz_list) / len(dxz_list))
        lyz.update(sum(dyz_list) / len(dyz_list))

        top1.update(prec1.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
        )
        pbar.set_description(msg)

    return loss_cls.avg, top1.avg, lxz.avg, lyz.avg


def test_attr(args, model, criterion, trainloader, a1, a2, n_classes, attr_num=20):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=140)
    for batch_idx, (inputs, _, attr) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)

        ##################
        iden = attr[attr_num]
        ##################

        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        loss, cross_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(args, model, criterion, inputs,
                                                                          iden, a1, a2, n_classes)

        # measure accuracy and record loss
        prec1, _ = accuracy(out_digit.data, iden.data, topk=(1, 1))
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(dxz_list) / len(dxz_list))
        lyz.update(sum(dyz_list) / len(dyz_list))

        top1.update(prec1.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
        )
        pbar.set_description(msg)

    return loss_cls.avg, top1.avg


def test_HSIC(args, model, criterion, testloader, a1, a2, n_classes):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    # pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=140)
    with torch.no_grad():
        for batch_idx, (inputs, iden) in enumerate(testloader):
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)

            loss, cross_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(args, model, criterion, inputs,
                                                                              iden, a1, a2, n_classes)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            loss_cls.update(cross_loss.item(), bs)
            lxz.update(sum(dxz_list) / len(dxz_list), bs)
            lyz.update(sum(dyz_list) / len(dyz_list), bs)

            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                # plot progress
                # print('CE {cls.val:.4f} ({cls.avg:.4f})\t'
                #     'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                #     'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                #     cls=loss_cls,
                #     loss=losses,
                #     top1=top1,
                # ))
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    batch_idx, len(testloader), batch_time=batch_time, loss=losses, loss_cls=loss_cls,
                    top1=top1))
                # print('(***Lxz(down) {lxz.val:.4f} ({lxz.avg:.4f})\t'
                #         'Lyz(up) {lyz.val:.4f} ({lyz.avg:.4f})***)\n'.format(
                #             lxz=lxz, lyz=lyz))

            # pbar.set_description(msg)

    print(' * Prec@1 {top1.avg:.3f}\n'.format(top1=top1))
    return loss_cls.avg, top1.avg


def train_reg(model, criterion, optimizer, trainloader):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100)

    for batch_idx, (inputs, iden) in pbar:
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, iden)

        loss = cross_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        bs = inputs.size(0)
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        loss_cls.update(cross_loss.item(), bs)

        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=loss_cls.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg


def test_reg(model, criterion, testloader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=100)

    ent_avg = 0
    prob_avg = 0
    with torch.no_grad():
        for batch_idx, (inputs, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)

            feats, out_digit = model(inputs)
            cross_loss = criterion(out_digit, iden)
            loss = cross_loss

            from utils import HLoss
            entropy = HLoss()
            ent_avg += entropy(out_digit)

            import torch.nn.functional as F
            prob = F.softmax(out_digit, dim=1)
            max_prob = torch.max(prob, 1)[0]
            prob_avg += sum(max_prob)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    # print("entropy:", ent_avg / len(testloader))
    # print("max_prob", prob_avg / len(testloader))
    return losses.avg, top1.avg


def train_vib(model, criterion, optimizer, trainloader, beta=1e-2):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100)

    for batch_idx, (inputs, targets) in pbar:
        # measure data loading time
        inputs, targets = inputs.cuda(), targets.cuda()
        bs = inputs.size(0)

        # compute output
        _, mu, std, out_digit = model(inputs)
        cross_loss = criterion(out_digit, targets)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=losses.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)
    return losses.avg, top1.avg


def test_vib(model, criterion, testloader, beta=1e-2):
    global best_acc

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=100)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            # measure data loading time

            inputs, targets = inputs.cuda(), targets.cuda()
            bs = inputs.size(0)

            # compute output
            _, mu, std, out_digit = model(inputs)
            cross_loss = criterion(out_digit, targets)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                cls=losses.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)
    return losses.avg, top1.avg


def multilayer_KD(args, teacher, student, criterion, inputs, target, a1, a2, b, n_classes, is_pretrained=False):
    dxz_list = []
    dyz_list = []
    bs = inputs.size(0)
    teacher_total_loss = 0
    student_total_loss = 0
    teacher_cross_loss = 0
    ####### teacher ###############
    teacher_hiddens, teacher_out_digit = teacher(inputs)
    if not is_pretrained:
        teacher_cross_loss = criterion(teacher_out_digit, target)

        teacher_total_loss += teacher_cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)
        for hidden in teacher_hiddens:
            hidden = hidden.view(bs, -1)

            if args.defense == 'HSIC':
                hxz_l, hyz_l = hsic.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=None,
                    ktype=args.ktype
                )
            elif args.defense == 'COCO':
                hxz_l, hyz_l = hsic.coco_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=None,
                    ktype=args.ktype
                )

            temp_hsic = a1 * hxz_l - a2 * hyz_l
            teacher_total_loss += temp_hsic

            # dxz_list.append(round(hxz_l.item(), 5))
            # dyz_list.append(round(hyz_l.item(), 5))

    #####################   student  ################################
    student_hiddens, student_out_digit = student(inputs)
    student_cross_loss = criterion(student_out_digit, target)

    student_total_loss += student_cross_loss
    h_target = utils.to_categorical(target, num_classes=n_classes).float()
    h_data = inputs.view(bs, -1)

    for hidden in student_hiddens:
        hidden = hidden.view(bs, -1)

        if args.defense == 'HSIC':
            hxz_l, hyz_l = hsic.hsic_objective(
                hidden,
                h_target=h_target.float(),
                h_data=h_data,
                sigma=None,
                ktype=args.ktype
            )
        elif args.defense == 'COCO':
            hxz_l, hyz_l = hsic.coco_objective(
                hidden,
                h_target=h_target.float(),
                h_data=h_data,
                sigma=None,
                ktype=args.ktype
            )

        dxz_list.append(round(hxz_l.item(), 5))
        dyz_list.append(round(hyz_l.item(), 5))

    KD_loss = 0
    for i in range(teacher_hiddens.__len__()):
        KD_loss += F.mse_loss(teacher_hiddens[i], student_hiddens[i])

    student_total_loss += b * KD_loss
    # return total_loss, cross_loss, out_digit, dxz_list, dyz_list  ################## modified by chj #################
    return teacher_total_loss, student_total_loss, teacher_cross_loss, student_cross_loss, KD_loss, teacher_out_digit, student_out_digit, dxz_list, dyz_list


def train_HSIC_gradmask(model, criterion, optimizer, trainloader, a1, a2, n_classes,
                        ktype='gaussian', hsic_training=True, defense='HSIC', masks=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        loss, cross_loss, hsic_loss, out_digit, dxz_list, dyz_list = multilayer_hsic(model, criterion, inputs, iden, a1,
                                                                                     a2,
                                                                                     n_classes, ktype, hsic_training,
                                                                                     defense)
        optimizer.zero_grad()
        loss.backward()
        # hsic_loss.backward()
        # cross_loss.backward()

        if masks is not None:
            for name, weight in model.named_parameters():
                if name not in masks: continue
                # weight_abs.append(torch.abs(weight))
                weight.grad = (1 - masks[name]) * weight.grad

        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item())
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))
        lxz.update(sum(dxz_list) / len(dxz_list))
        lyz.update(sum(dyz_list) / len(dyz_list))

        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg


def train_KD(args, teacher, student, criterion, teacher_optimizer, student_optimizer, trainloader, a1, a2, b, n_classes,
             masks=None, is_pretrained=True):
    if is_pretrained:
        teacher.eval()
    else:
        teacher.train()

    student.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    loss_kd = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        teacher_loss, student_loss, teacher_cross_loss, student_cross_loss, KD_loss, teacher_out_digit, student_out_digit, dxz_list, dyz_list = \
            multilayer_KD(args, teacher, student, criterion, inputs, iden, a1, a2, b, n_classes, is_pretrained)

        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()

        # if not is_pretrained:
        #     teacher_optimizer.zero_grad()
        #     teacher_loss.backward()
        #     teacher_optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(student_out_digit.data, iden.data, topk=(1, 5))
        losses.update(student_loss.item())
        loss_cls.update(float(student_cross_loss.detach().cpu().numpy()))
        loss_kd.update(KD_loss.item())
        lxz.update(sum(dxz_list) / len(dxz_list))
        lyz.update(sum(dyz_list) / len(dyz_list))

        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | KD:{kd:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            cls=loss_cls.avg,
            kd=loss_kd.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg, lxz.avg, lyz.avg

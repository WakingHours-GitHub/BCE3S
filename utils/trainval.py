import os
import time
import copy
import numpy as np
import math
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import nni


def train_model(dataloaders, model, lossFunc, optimizerW, schedulerW, scaler=None, 
                pgdFunc=None, CB_loss=None, num_epochs=50, model_name='model', 
                work_dir='./', device='cpu', freqShow=40, clipValue=1, print_each=1, 
                writer: SummaryWriter=None):
    """
    Train model with standard training loop
    """
    train_counter = 1
    val_counter = 1
    lr_counter = 1
    
    trackRecords = {
        'weightNorm': [],
        'acc_test': [],
        'acc_train': [],
        'weights': []
    }
    
    log_filename = os.path.join(work_dir, model_name + '_train.log')
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.0
    best_perClassAcc = 0.0
    
    phaseList = list(dataloaders.keys())
    phaseList.remove('train')
    phaseList = ['train'] + phaseList
    
    for epoch in range(num_epochs):
        if epoch % print_each == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 10)
        
        with open(log_filename, 'a') as fn:
            fn.write(f'\nEpoch {epoch+1}/{num_epochs}\n')
            fn.write('--' * 5 + '\n')

        for phase in phaseList:
            if epoch % print_each == 0:
                print(phase)
            
            predList = np.array([])
            grndList = np.array([])
            
            with open(log_filename, 'a') as fn:
                fn.write(phase + '\n')
            
            if phase == 'train':
                schedulerW.step()
                model.train()
                lossFunc.train()
                if writer is not None:
                    writer.add_scalar("learn_rate", optimizerW.param_groups[0]['lr'], lr_counter)
                lr_counter += 1
            else:
                model.eval()
                lossFunc.eval()
                
            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0
            
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:
                imageList, labelList = sample
                imageList = imageList.to(device)
                labelList = labelList.view(-1).to(device)

                optimizerW.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            features = model(imageList)
                            loss, logits = lossFunc(features, labelList, epoch)
                            if CB_loss:
                                loss = CB_loss(logits, labelList.type(torch.long))
                    else:
                        features = model(imageList)
                        loss, logits = lossFunc(features, labelList, epoch)
                        if CB_loss:
                            loss = CB_loss(logits, labelList.type(torch.long))
                    
                    if "distance" in lossFunc.__class__.__name__:
                        predLabel = logits.argmax(dim=1).detach().squeeze().type(torch.float)
                        accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                        accRate = (accRate == 0).type(torch.float).mean()
                    else:
                        softmaxScores = logits
                        predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)
                        accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                        accRate = (accRate == 0).type(torch.float).mean()
                    
                    predList = np.concatenate((predList, predLabel.cpu().numpy()))
                    grndList = np.concatenate((grndList, labelList.cpu().numpy()))

                    if phase == 'train':
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizerW)
                            scaler.step(optimizerW)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizerW.step()
                iterCount += 1
                sampleCount += labelList.size(0)
                running_acc += accRate * labelList.size(0)
                running_loss_CE += loss.item() * labelList.size(0)
                running_loss = running_loss_CE
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                
            epoch_error = print2screen_avgLoss
            
            confMat = sklearn.metrics.confusion_matrix(grndList, predList)
            a = confMat.sum(axis=1).reshape((-1, 1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i, i]
            curPerClassAcc /= confMat.shape[0]
            
            if epoch % print_each == 0:
                if phase == 'train':
                    if writer is not None:
                        writer.add_scalar("Train_loss", epoch_error, train_counter)
                    train_counter += 1
                else:
                    if writer is not None:
                        writer.add_scalar("Val_loss", epoch_error, val_counter)
                    val_counter += 1
                print(f'\tloss:{epoch_error:.6f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{curPerClassAcc:.5f} | lr: {optimizerW.param_groups[0]["lr"]}')

            with open(log_filename, 'a') as fn:
                fn.write(f'\tloss:{epoch_error:.6f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{curPerClassAcc:.5f}')
                
            if phase == 'train':
                if pgdFunc:
                    pgdFunc.PGD(lossFunc)
                trackRecords['acc_train'].append(curPerClassAcc)
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                W = lossFunc.weight.cpu().clone()
                tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
                trackRecords['weightNorm'].append(tmp)
                trackRecords['weights'].append(W.detach().cpu().numpy())
                
            if (phase == 'val' or phase == 'test'):
                nni.report_intermediate_result(curPerClassAcc)
            
            if (phase == 'val' or phase == 'test') and curPerClassAcc > best_perClassAcc:
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name + '_best.paramOnly')
                torch.save(model.state_dict(), path_to_save_param)
                
                path_to_save_param_classifier = os.path.join(work_dir, model_name + '_classifier_best.paramOnly')
                torch.save(lossFunc.state_dict(), path_to_save_param_classifier)
                
                file_to_note_bestModel = os.path.join(work_dir, model_name + '_note_bestModel.log')
                with open(file_to_note_bestModel, 'a') as fn:
                    fn.write(f'The best model is achieved at epoch-{epoch+1}: loss{best_loss:.5f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{best_perClassAcc:.5f}.\n')
    
    path_to_save_param = os.path.join(work_dir, model_name + '_last.paramOnly')
    torch.save(model.state_dict(), path_to_save_param)
    
    path_to_save_param_classifier = os.path.join(work_dir, model_name + '_classifier_last.paramOnly')
    torch.save(lossFunc.state_dict(), path_to_save_param_classifier)
    
    time_elapsed = time.time() - since
    trackRecords['time_elapsed'] = time_elapsed
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    with open(log_filename, 'a') as fn:
        fn.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    
    return trackRecords


def train_contrastive_learning_model(dataloaders, weight_dataloaders, model, lossFunc, 
                                   optimizerW, schedulerW, scaler=None, pgdFunc=None, CB_loss=None, 
                                   num_epochs=50, model_name='model', work_dir='./', device='cpu', 
                                   freqShow=40, clipValue=1, print_each=1, clip_grad=10.0, 
                                   writer: SummaryWriter=None, n_class_num=None):
    """
    Train model with contrastive learning using GLMC mixing
    """
    train_counter = 1
    val_counter = 1
    lr_counter = 1
    
    trackRecords = {
        'weightNorm': [],
        'acc_test': [],
        'acc_train': [],
        'weights': []
    }
    
    log_filename = os.path.join(work_dir, model_name + '_train.log')
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.0
    best_perClassAcc = 0.0
    
    phaseList = list(dataloaders.keys())
    phaseList.remove('train')
    phaseList = ['train'] + phaseList
    
    for epoch in range(num_epochs):
        if epoch % print_each == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
        with open(log_filename, 'a') as fn:
            fn.write(f'\nEpoch {epoch+1}/{num_epochs}\n')
            fn.write('--' * 5 + '\n')

        weight_dataloaders_epoch = iter(weight_dataloaders)
        
        for phase in phaseList:
            if epoch % print_each == 0:
                print(phase)
            
            predList = np.array([])
            grndList = np.array([])
            
            with open(log_filename, 'a') as fn:
                fn.write(phase + '\n')
            
            if phase == 'train':
                schedulerW.step()
                model.train()
                lossFunc.train()
                if writer is not None:
                    writer.add_scalar("learn_rate", optimizerW.param_groups[0]['lr'], lr_counter)
                lr_counter += 1
            else:
                model.eval()
                lossFunc.eval()
                
            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0
            
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:
                if phase == "train":
                    try:
                        input_invs, label_invs = next(weight_dataloaders_epoch)
                    except:
                        weight_dataloaders_epoch = iter(weight_dataloaders)
                        input_invs, label_invs = next(weight_dataloaders_epoch)

                    num_classes = n_class_num if n_class_num else lossFunc.weight.size(0)
                    input_invs, label_invs = input_invs.to(device), label_invs.view(-1).to(device)
                    one_hot_invs = torch.zeros(label_invs.size(0), num_classes, device=device).scatter_(1, label_invs.view(-1, 1).long(), 1)
                
                    input_origin, label_origin = sample
                    input_origin, label_origin = input_origin.to(device), label_origin.view(-1).to(device)
                    one_hot_origin = torch.zeros(label_origin.size(0), num_classes, device=device).scatter_(1, label_origin.view(-1, 1).long(), 1)
                        
                    source, mixup_input, cutmix_input, mixup_onehot, cutmix_onehot = GLMC_mixed(input_origin, input_invs, one_hot_origin, one_hot_invs)
                else:
                    input_origin, label_origin = sample
                    input_origin, label_origin = input_origin.to(device), label_origin.view(-1).to(device)
                    
                optimizerW.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            if phase == "train":
                                concatenated_inputs = torch.cat((source, mixup_input, cutmix_input), dim=0)
                                features = model(concatenated_inputs)
                                loss, logits = lossFunc(features, label_origin, mixup_onehot, cutmix_onehot, epoch, phase='train')
                            else:
                                features = model(input_origin)
                                loss, logits = lossFunc(features, label_origin)
                    else:
                        if phase == "train":
                            concatenated_inputs = torch.cat((source, mixup_input, cutmix_input), dim=0)
                            features = model(concatenated_inputs)
                            loss, logits = lossFunc(features, label_origin, mixup_onehot, cutmix_onehot, epoch, phase='train')
                        else:
                            features = model(input_origin)
                            loss, logits = lossFunc(features, label_origin)

                    softmaxScores = logits
                    predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)
                    accRate = (label_origin.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                    accRate = (accRate == 0).type(torch.float).mean()
                    
                    predList = np.concatenate((predList, predLabel.cpu().numpy()))
                    grndList = np.concatenate((grndList, label_origin.cpu().numpy()))

                    if phase == 'train':
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizerW)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                            torch.nn.utils.clip_grad_norm_(lossFunc.parameters(), max_norm=clip_grad)
                            scaler.step(optimizerW)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizerW.step()
                iterCount += 1
                sampleCount += label_origin.size(0)
                running_acc += accRate * label_origin.size(0)
                running_loss_CE += loss.item() * label_origin.size(0)
                running_loss = running_loss_CE
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                
            epoch_error = print2screen_avgLoss
            
            confMat = sklearn.metrics.confusion_matrix(grndList, predList)
            a = confMat.sum(axis=1).reshape((-1, 1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i, i]
            curPerClassAcc /= confMat.shape[0]
            
            if epoch % print_each == 0:
                if phase == 'train':
                    if writer is not None:
                        writer.add_scalar("Train_loss", epoch_error, train_counter)
                    train_counter += 1
                else:
                    if writer is not None:
                        writer.add_scalar("Val_loss", epoch_error, val_counter)
                    val_counter += 1
                print(f'\tloss:{epoch_error:.6f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{curPerClassAcc:.5f} | lr: {optimizerW.param_groups[0]["lr"]}')

            with open(log_filename, 'a') as fn:
                fn.write(f'\tloss:{epoch_error:.6f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{curPerClassAcc:.5f}')
                
            if phase == 'train':
                if pgdFunc:
                    pgdFunc.PGD(lossFunc)
                trackRecords['acc_train'].append(curPerClassAcc)
            else:
                trackRecords['acc_test'].append(curPerClassAcc)
                
            if (phase == 'val' or phase == 'test'):
                nni.report_intermediate_result(curPerClassAcc)
            
            if (phase == 'val' or phase == 'test') and curPerClassAcc > best_perClassAcc:
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name + '_best.paramOnly')
                torch.save(model.state_dict(), path_to_save_param)
                
                path_to_save_param_classifier = os.path.join(work_dir, model_name + '_classifier_best.paramOnly')
                torch.save(lossFunc.state_dict(), path_to_save_param_classifier)
                
                file_to_note_bestModel = os.path.join(work_dir, model_name + '_note_bestModel.log')
                with open(file_to_note_bestModel, 'a') as fn:
                    fn.write(f'The best model is achieved at epoch-{epoch+1}: loss{best_loss:.5f}, acc-all:{print2screen_avgAccRate:.5f}, acc-avg-cls:{best_perClassAcc:.5f}.\n')
                
    time_elapsed = time.time() - since
    trackRecords['time_elapsed'] = time_elapsed
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    path_to_save_param = os.path.join(work_dir, model_name + '_last.paramOnly')
    torch.save(model.state_dict(), path_to_save_param)
    
    path_to_save_param_classifier = os.path.join(work_dir, model_name + '_classifier_last.paramOnly')
    torch.save(lossFunc.state_dict(), path_to_save_param_classifier)
    
    with open(log_filename, 'a') as fn:
        fn.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    
    return trackRecords


def save_checkpoint(model, loss_head, optimizer, scheduler, scaler, epoch, best_acc, 
                    is_best=False, work_dir='./', model_name='model', rank=0):
    """Save training checkpoint"""
    state = {
        'model': model.state_dict(),
        'loss_head': loss_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'best_acc': best_acc
    }
    
    checkpoint_path = os.path.join(work_dir, f"{model_name}_checkpoint_latest.pth")
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(work_dir, f"{model_name}_checkpoint_best.pth")
        torch.save(state, best_path)
        print(f"Best model saved, accuracy: {best_acc:.4f}, epoch: {epoch}, path: {best_path}")


def print_on_rank0(message, rank=0):
    """Print message only on rank 0"""
    if rank == 0:
        print(message)


def get_feature_mean(imbalanced_train_loader, model, cls_num_list, embedding=512):
    """Calculate feature means for each class"""
    model.eval()
    cls_num = len(cls_num_list)
    feature_mean_end = torch.zeros(cls_num, embedding).cuda()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda().view(-1).long()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            features = model(input_var)
            features = features.detach()
            features = features.cpu().data.numpy()
            
            for out, label in zip(features, target):
                feature_mean_end[label] = feature_mean_end[label] + torch.tensor(out).cuda()

        img_num_list_tensor = torch.tensor(cls_num_list).unsqueeze(1).cuda()
        feature_mean_end = torch.div(feature_mean_end, img_num_list_tensor).detach()
        return feature_mean_end


def to_var(x, requires_grad=True):
    """Convert tensor to variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def calculate_eff_weight(imbalanced_train_loader, model, cls_num_list, train_propertype):
    """Calculate effective weights for classes"""
    model.eval()
    train_propertype = train_propertype.cuda()
    class_num = len(cls_num_list)
    eff_all = torch.zeros(class_num).float().cuda()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda().view(-1).long()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            features = model(input_var)
            mu = train_propertype[target.long()].detach()
            feature_bz = (features.detach() - mu)
            index = torch.unique(target)
            index2 = target.cpu().numpy()
            eff = torch.zeros(class_num).float().cuda()
            
            for i in range(len(index)):
                index3 = torch.from_numpy(np.argwhere(index2 == index[i].item()))
                index3 = torch.squeeze(index3)
                feature_matrix = feature_bz[index3].detach()
                
                if feature_matrix.dim() == 1:
                    eff[index[i]] = 1
                else:
                    matA_matB = torch.matmul(feature_matrix, feature_matrix.transpose(0, 1))
                    matA_norm = torch.unsqueeze(torch.sqrt(torch.mul(feature_matrix, feature_matrix).sum(axis=1)), 1)
                    matA_matB_length = torch.mul(matA_norm, matA_norm.transpose(0, 1))
                    matA_matB_length[matA_matB_length == 0] = 1
                    r = torch.div(matA_matB, matA_matB_length)
                    num = feature_matrix.size(0)
                    a = (torch.ones(1, num).float().cuda()) / num
                    b = (torch.ones(num, 1).float().cuda()) / num
                    c = torch.matmul(torch.matmul(a, r), b).float().cuda()
                    eff[index[i]] = 1 / c
            eff_all = eff_all + eff
            
        weights = eff_all
        weights = torch.where(weights > 0, 1 / weights, weights).detach()
        fen_mu = torch.sum(weights)
        weights_new = weights / fen_mu
        weights_new = weights_new * class_num
        weights_new = weights_new.detach()
        return weights_new


def compute_uniform_classification_accuracy(softmaxScores, labelList):
    """Compute uniform classification accuracy with threshold optimization"""
    best_accuracy = 0.0
    best_threshold = 0.0
    for threshold in torch.linspace(0, 1, steps=1000):
        predictions = (softmaxScores > threshold).float()
        correct_predictions = (predictions.argmax(dim=1) == labelList).float()
        accuracy = correct_predictions.mean().item()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_accuracy, best_threshold


def adjust_learning_rate(optimizer, epoch, lr, mode="multistep", num_epochs=200, step_lr=[160, 180]):
    """Adjust learning rate based on schedule"""
    if mode == 'cos':
        lr_min = 0
        lr_max = lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / num_epochs * 3.1415926535))
    elif mode == 'multistep':
        epoch = epoch + 1
        if epoch <= 5:
            lr = lr * epoch / 5
        elif epoch > step_lr[1]:
            lr = lr * 0.01
        elif epoch > step_lr[0]:
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def GLMC_mixed(org, invs, onehot_org, onehot_invs, alpha=1):
    """GLMC mixing for contrastive learning"""
    lam = np.random.beta(alpha, alpha)
    source = org.clone()
    
    # Mixup
    mixup_x = lam * org + (1 - lam) * invs
    mixup_y = lam * onehot_org + (1 - lam) * onehot_invs

    # CutMix
    bbx1, bby1, bbx2, bby2 = rand_bbox(org.size(), lam)
    org[:, :, bbx1:bbx2, bby1:bby2] = invs[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * onehot_org + (1 - lam_cutmix) * onehot_invs

    return source, mixup_x, org, mixup_y, cutmix_y


def GLMC_mixed_ddp(org, invs, onehot_org, onehot_invs, alpha=1):
    """GLMC mixing function supporting DDP mode"""
    batch_size = min(org.size(0), invs.size(0), onehot_org.size(0), onehot_invs.size(0))
    if org.size(0) > batch_size:
        org = org[:batch_size]
    if invs.size(0) > batch_size:
        invs = invs[:batch_size]
    if onehot_org.size(0) > batch_size:
        onehot_org = onehot_org[:batch_size]
    if onehot_invs.size(0) > batch_size:
        onehot_invs = onehot_invs[:batch_size]
        
    lam = np.random.beta(alpha, alpha)
    source = org.clone()
    mixup_x = lam * org + (1 - lam) * invs
    mixup_y = lam * onehot_org + (1 - lam) * onehot_invs
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(org.size(), lam)
    org_cut = org.clone()
    org_cut[:, :, bbx1:bbx2, bby1:bby2] = invs[:, :, bbx1:bbx2, bby1:bby2]
    
    lam_cutmix = lam
    cutmix_y = lam_cutmix * onehot_org + (1 - lam_cutmix) * onehot_invs
    
    return source, mixup_x, org_cut, mixup_y, cutmix_y


def print_model_param_nums(model=None):
    """Print number of model parameters"""
    total = sum([param.nelement() for param in model.parameters()])
    print(f'Number of params: {total / 1e6:.2f}M')
    return total


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(np.ceil(W * cut_rat))
    cut_h = int(np.ceil(H * cut_rat))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

import os, sys, random, copy
import numpy as np
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import wandb

from metrics.accuracy import Accuracy
from torch.utils.data import DataLoader
from torchvision.models import densenet, inception, resnet
# import torchvision.models.vision_transformer as vits

from tqdm import tqdm
from copy import deepcopy
from torchvision.transforms import (ColorJitter, RandomCrop,
                                    RandomHorizontalFlip, Resize)
from datasets.custom_subset import Subset
from collections import defaultdict
from models.base_model import BaseModel
from models.resnet import resnet18, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from models.densenet import densenet121
from models.resnest import resnest50
from models.vgg import vgg16_bn, VGG16
from utils.hsic import hsic_objective, coco_objective
from utils.logger import CSVLogger, plot_csv, Tee
from torch.utils.data import sampler



class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)

            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_feature_extractor(net, state_dict):
    print("load_pretrained_feature_extractor!!!")
    net_state = net.state_dict()

    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        if "running_var" in name:
            new_state_dict[name] = param
            new_item = name.replace("running_var", "num_batches_tracked")
            new_state_dict[new_item] = torch.tensor(0)
        else:
            new_state_dict[name] = param

    for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
        if "classifier" in new_name:
            break
        if "num_batches_tracked" in new_name:
            continue

        net_state[name].copy_(mew_param.data)


def bilateral_dependency_loss(inputs, hiddens, labels, num_classes, measure, alpha, beta):
    dxz_list = []
    dyz_list = []

    bs = inputs.size(0)

    bido_dxz_list = []
    bido_dyz_list = []

    h_target = F.one_hot(labels, num_classes).float()
    h_data = inputs.view(bs, -1)

    bido_loss = 0
    for hidden in hiddens:
        hidden = hidden.view(bs, -1)

        if measure == 'HSIC':
            hxz_l, hyz_l = hsic_objective(
                hidden,
                h_target=h_target,
                h_data=h_data,
                sigma=5,
                ktype='linear'
            )

        elif measure == 'COCO':
            hxz_l, hyz_l = coco_objective(
                hidden,
                h_target=h_target,
                h_data=h_data,
                sigma=5,
                ktype='linear'
            )

        dxz_list.append(hxz_l.item())
        dyz_list.append(hyz_l.item())

        bido_dxz_list.append(hxz_l)
        bido_dyz_list.append(hyz_l)

    for (hxz_l, hyz_l) in zip(bido_dxz_list, bido_dyz_list):
        temp_hsic = alpha * hxz_l - beta * hyz_l
        bido_loss += temp_hsic

    return bido_loss, dxz_list, dyz_list

def select_ood(mode, ood_factor, trainloader, aux_ood_set, model, device, batch_size, dataloader_num_workers, bayes_nn=None):
    if mode == "random_OE":
        indices = list(range(len(aux_ood_set)))
        oodloader = DataLoader(
            Subset(aux_ood_set, indices),
            batch_size=int(batch_size * ood_factor),
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=True
        )
        oodloader.dataset.offset = np.random.randint(len(oodloader.dataset))

        return oodloader

    ############################ hyper-parameters ############################
    indices = list(range(len(aux_ood_set)))
    oodloader = DataLoader(
        Subset(aux_ood_set, indices),
        batch_size=1024,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    quantile = 0.25
    ood_bs = int(batch_size * ood_factor)
    ood_dataset_size = ood_bs * len(trainloader)
    ############################# hyper-parameters ###########################

    if mode == "NTOM":
        model.eval()
        with torch.no_grad():
            all_ood_conf = []
            for ood_inputs, labels in tqdm(oodloader,
                                           desc='ood selection',
                                           leave=False,
                                           file=sys.stdout):
                ood_inputs = ood_inputs.to(device, non_blocking=True)

                ood_outputs = model(ood_inputs)
                if isinstance(ood_outputs, tuple):
                    ood_outputs = ood_outputs[-1]

                ood_conf = F.softmax(ood_outputs, dim=1)[:, -1]
                # select by ood_conf
                all_ood_conf.extend(ood_conf.detach().cpu().numpy())

        all_ood_conf = np.array(all_ood_conf)
        indices = np.argsort(all_ood_conf)

        aux_ood_size = len(indices)
        selected_indices = indices[int(quantile * aux_ood_size):int(quantile * aux_ood_size) + ood_dataset_size]

    elif mode == "POEM":
        model.eval()
        with torch.no_grad():
            all_abs_val = torch.empty(0)
            all_ood_conf = []
            for ood_inputs, labels in tqdm(oodloader,
                                           desc='ood selection',
                                           leave=False,
                                           file=sys.stdout):
                ood_inputs = ood_inputs.to(device, non_blocking=True)
                output = bayes_nn.predict(ood_inputs)
                abs_val = torch.abs(output).squeeze()
                all_abs_val = torch.cat((all_abs_val, abs_val.detach().cpu()), dim=0)

                ood_outputs = model(ood_inputs)
                if isinstance(ood_outputs, tuple):
                    ood_outputs = ood_outputs[-1]

                ood_conf = F.softmax(ood_outputs, dim=1)[:, -1]
                # select by ood_conf
                all_ood_conf.extend(ood_conf.detach().cpu().numpy())

        all_ood_conf = np.array(all_ood_conf)
        _, selected_indices = torch.topk(all_abs_val, ood_dataset_size, largest=False)

    ########################################### log & return #############################################
    print()
    print('Min OOD Conf: ', np.min(all_ood_conf), 'Max OOD Conf: ', np.max(all_ood_conf), 'Average OOD Conf: ',
          np.mean(all_ood_conf))
    print('Selected OOD samples: ', len(selected_indices))
    selected_ood_conf = all_ood_conf[selected_indices]
    print('Selected Min OOD Conf: ', np.min(selected_ood_conf), 'Selected Max OOD Conf: ', np.max(selected_ood_conf),
          'Selected Average OOD Conf: ', np.mean(selected_ood_conf))
    print()

    selected_oodloader = DataLoader(
        Subset(aux_ood_set, selected_indices),
        batch_size=ood_bs,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True
    )

    return selected_oodloader
    ########################################### log & return #############################################


class Classifier(BaseModel):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 architecture='resnet18',
                 pretrained=False,
                 bido_args=None,
                 OE_args=None,
                 name='Classifier',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.bido_args = bido_args
        self.OE_args = OE_args
        self.model = self._build_model(architecture, pretrained)
        self.model.to(self.device)
        self.architecture = architecture

        self.to(self.device)

    def get_representation(self, x):
        return self.model.embedding(x)
    

    def _build_model(self, architecture, pretrained):
        architecture = architecture.lower().replace('-', '').strip()
        if 'vgg' in architecture:
            model = vgg16_bn(pretrained=pretrained)
            if self.bido_args['enable_bido']:
                model = VGG16(bido=True)
                checkpoint = torch.load(
                    "./results/pretrained/vgg16_bn-6c64b313.pth")
                load_feature_extractor(model, checkpoint)
            else:
                model = VGG16()  

        elif 'resnet' in architecture:
            if architecture == 'resnet18':
                model = resnet18(pretrained=pretrained)
            elif architecture == 'resnet101':
                model = resnet101(pretrained=pretrained)
            elif architecture == 'resnet152':
                model = resnet152(pretrained=pretrained)

        elif 'resnext' in architecture:
            if architecture == 'resnext50':
                model = resnext50_32x4d(pretrained=pretrained)
            elif architecture == 'resnext101':
                model = resnext101_32x8d(pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No ResNext with the name {architecture} available')

        elif 'resnest' in architecture:
            if architecture == 'resnest50':
                model = resnest50(pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No ResNeSt with the name {architecture} available')

        elif 'densenet' in architecture:
            # if self.bido_args['enable_bido']:
            if architecture == 'densenet121':
                model = densenet121(pretrained=pretrained)

            if self.num_classes != model.classifier.out_features:
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, self.num_classes)

        # Note: inception_v3 expects input tensors with a size of N x 3 x 299 x 299, aux_logits are used per default
        elif 'inception' in architecture:
            model = inception.inception_v3(pretrained=pretrained,
                                           aux_logits=True,
                                           init_weights=True)
            if self.num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        else:
            raise RuntimeError(
                f'No network with the name {architecture} available')

        return model

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        out = self.model(x)
        return out

    def fit(self,
            training_data,
            validation_data=None,
            test_data=None,
            optimizer=None,
            lr_scheduler=None,
            criterion=nn.CrossEntropyLoss(),
            metric=Accuracy,
            rtpt=None,
            config=None,
            batch_size=64,
            num_epochs=30,
            dataloader_num_workers=8,
            enable_logging=False,
            save_base_path="",
            config_file=None):

        sampler = RandomIdentitySampler(training_data, batch_size, num_instances=4)


        trainloader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 sampler=sampler,
                                 num_workers=dataloader_num_workers,
                                 pin_memory=True)

        if rtpt is None:
            print('Please use RTPT (Remaining Time to Process Title)')


        # Training cycle
        best_model_values = {
            'validation_metric': 0.0,
            'validation_loss': float('inf'),
            'model_state_dict': None,
            'model_optimizer_state_dict': None,
            'training_metric': 0,
            'training_loss': 0,
        }

        epoch_fieldnames = ['global_iteration', 'train_acc', 'test_acc']
        epoch_logger = CSVLogger(every=1,
                                 fieldnames=epoch_fieldnames,
                                 filename=os.path.join(save_base_path, f'epoch_log.csv'),
                                 resume=0)
        Tee(os.path.join(save_base_path, "log.txt"), mode='w')

        metric_train = metric()

        print('----------------------- START TRAINING -----------------------')
        for epoch in range(num_epochs):
            # Training
            print(f'Epoch {epoch + 1}/{num_epochs}')
            running_total_loss = 0.0
            running_main_loss = 0.0
            running_aux_loss = 0.0
            running_bido_loss = 0.0
            running_id_conf = 0.0

            metric_train.reset()
            self.to(self.device)
            self.train()

            for batch_idx, (inputs, labels) in enumerate(
                    tqdm(trainloader, desc='training', leave=False, file=sys.stdout)):
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                aux_loss = torch.tensor(0.0, device=self.device)
                bido_loss = torch.tensor(0.0, device=self.device)

                num_samples = inputs.shape[0]

                if self.bido_args['enable_bido']:
                    hiddens, model_output = self.forward(inputs)
                    main_loss = criterion(model_output, labels)

                    bido_loss, dxz_list, dyz_list = bilateral_dependency_loss(inputs, hiddens, labels,
                                                          num_classes=self.num_classes,
                                                          measure=self.bido_args["params"]["measure"],
                                                          alpha=self.bido_args["params"]["alpha"],
                                                          beta=self.bido_args["params"]["beta"])
                    loss = main_loss + bido_loss
                else:
                    model_output = self.forward(inputs)
                    if type(model_output) is tuple:
                        model_output = model_output[-1]

                    # Separate Inception_v3 outputs
                    aux_logits = None
                    if isinstance(model_output, inception.InceptionOutputs):
                        if self.model.aux_logits:
                            model_output, aux_logits = model_output

                    main_loss = criterion(model_output, labels)
                    if aux_logits is not None:
                        aux_loss += criterion(aux_logits, labels).sum()

                    loss = main_loss + aux_loss

                id_conf = F.softmax(model_output, dim=1)
                mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()

                loss.backward()
                optimizer.step()

                running_total_loss += loss * num_samples
                running_main_loss += main_loss * num_samples
                running_aux_loss += aux_loss * num_samples
                running_bido_loss += bido_loss * num_samples
                running_id_conf += mean_id_conf * num_samples

                metric_train.update(model_output, labels)

            print(
                f'Training {metric_train.name}:   {metric_train.compute_metric():.2%}',
                f'\t Epoch total loss: {running_total_loss / len(training_data):.4f}',
                f'\t Epoch main loss: {running_main_loss / len(training_data):.4f}',
                f'\t bido loss: {running_bido_loss / len(training_data):.4f}',
                f'\t id conf: {running_id_conf / len(training_data):.4f}',
            )

            # Validation
            if validation_data:
                self.eval()
                val_metric, val_loss, running_id_conf = self.evaluate(
                    validation_data,
                    batch_size,
                    metric,
                    criterion,
                    dataloader_num_workers=dataloader_num_workers)

                ################################################################################################
                epoch_logger.writerow({
                    'global_iteration': epoch,
                    'train_acc': metric_train.compute_metric(),
                    'test_acc': val_metric,

                })
                plot_csv(epoch_logger.filename, os.path.join(save_base_path, f'epoch_plots.jpeg'))
                ################################################################################################

                print(
                    f'Validation {metric_train.name}: {val_metric:.2%}',
                    f'\t Validation Loss:  {val_loss:.4f}',
                    f'\t id conf: {running_id_conf / len(validation_data):.4f}',
                )

                # Save best model
                if val_metric > best_model_values['validation_metric']:
                    print('Copying better model')
                    best_model_values['validation_metric'] = val_metric
                    best_model_values['validation_loss'] = val_loss
                    best_model_values['model_state_dict'] = deepcopy(self.state_dict())
                    best_model_values['model_optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                    best_model_values['training_metric'] = metric_train.compute_metric()
                    best_model_values['training_loss'] = running_total_loss / len(trainloader)


            else:
                best_model_values['validation_metric'] = None
                best_model_values['validation_loss'] = None
                best_model_values['model_state_dict'] = deepcopy(self.state_dict())
                best_model_values['model_optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                best_model_values['training_metric'] = metric_train.compute_metric()
                best_model_values['training_loss'] = running_total_loss / len(trainloader)

            # Update the RTPT
            rtpt.step(subtitle=f"loss={running_total_loss / len(trainloader):.4f}")

            # make the lr scheduler step
            if lr_scheduler is not None:
                lr_scheduler.step()

        # save the final model
        if validation_data:
            self.load_state_dict(best_model_values['model_state_dict'])

        if save_base_path:
            if not os.path.exists(save_base_path):
                os.makedirs(save_base_path)
            if validation_data:
                model_path = os.path.join(
                    save_base_path, self.name + f'_{best_model_values["validation_metric"]:.4f}' + '.pth')
            else:
                model_path = os.path.join(
                    save_base_path, self.name + f'_{best_model_values["training_metric"]:.4f}_no_val' + '.pth')

        else:
            model_path = self.name

        torch.save(
            {
                'epoch': num_epochs,
                'model_state_dict': best_model_values['model_state_dict'],
                'optimizer_state_dict': best_model_values['model_optimizer_state_dict'],
            }, model_path)

        val_metric = best_model_values['validation_metric']
        val_loss = best_model_values['validation_loss']

        print(
            f'\nBest Val {metric_train.name}: {val_metric:.2%} \t Val Loss: {val_loss:.4f} \n'
        )
        # Test final model
        test_metric, test_loss = None, None
        if test_data:
            test_metric, test_loss, _ = self.evaluate(
                test_data,
                batch_size,
                metric,
                criterion,
                dataloader_num_workers=dataloader_num_workers)
            print(
                '----------------------- FINISH TRAINING -----------------------'
            )
            print(
                f'Final Test {metric_train.name}: {test_metric:.2%} \t Test Loss: {test_loss:.4f} \n'
            )


    def OE_fit(self,
               training_data,
               validation_data=None,
               test_data=None,
               ood_data=None,
               optimizer=None,
               lr_scheduler=None,
               criterion=nn.CrossEntropyLoss(),
               metric=Accuracy,
               rtpt=None,
               config=None,
               batch_size=64,
               num_epochs=30,
               dataloader_num_workers=8,
               enable_logging=False,
               save_base_path="",
               config_file=None):

        trainloader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=dataloader_num_workers,
                                 pin_memory=True)

        # Training cycle
        best_model_values = {
            'validation_metric': 0.0,
            'validation_loss': float('inf'),
            'model_state_dict': None,
            'model_optimizer_state_dict': None,
            'training_metric': 0,
            'training_loss': 0,
            'epoch': -1
        }

        epoch_fieldnames = ['global_iteration', 'train_acc', 'test_acc']
        epoch_logger = CSVLogger(every=1,
                                 fieldnames=epoch_fieldnames,
                                 filename=os.path.join(save_base_path, f'epoch_log.csv'),
                                 resume=0)
        Tee(os.path.join(save_base_path, "log.txt"), mode='w')
        metric_train = metric()

        print('----------------------- START TRAINING -----------------------')

        num_training_data = len(trainloader.dataset)
        ood_factor = self.OE_args['params']['ood_factor']

        m_in = 0
        m_out = 0

        ft_epochs = 60  
        gamma = 1e-3

        for epoch in range(num_epochs):
            oodloader = select_ood(
                self.OE_args['mode'],
                ood_factor,
                trainloader,
                ood_data,
                self.model,
                self.device,
                batch_size,
                dataloader_num_workers)

            out_energy_losses = AverageMeter()
            in_energy_losses = AverageMeter()

            # Training
            running_total_loss = 0.0
            running_id_loss = 0.0
            running_id_conf = 0.0
            running_bido_loss = 0.0

            metric_train.reset()
            self.to(self.device)
            self.train()
            print(f'Epoch {epoch + 1}/{num_epochs}')

            if epoch < ft_epochs - 1:
                continue
            
            elif epoch == ft_epochs - 1:

                checkpoint_path = "./results/facescrub_densenet121/20250304_172411_HSIC_0_0_OE__celeba_identities/Classifier_0.9653.pth"
                checkpoint = torch.load(checkpoint_path)

                # Load the model and optimizer state
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for i, (id_batch, ood_batch) in enumerate(tqdm(zip(trainloader, oodloader), ncols=150,
                                                           total=len(trainloader), desc='training', leave=False,
                                                           file=sys.stdout)):
                id_bs = len(id_batch[0])
                ood_bs = len(ood_batch[0])

                id_inputs, id_labels = id_batch[0].to(self.device, non_blocking=True), id_batch[1].to(self.device,
                                                                                                      non_blocking=True)
                ood_inputs, _ = ood_batch[0].to(self.device, non_blocking=True), ood_batch[1].to(self.device,
                                                                                                          non_blocking=True)


                bido_loss = torch.tensor(0.0, device=self.device)

                if self.bido_args['enable_bido']:
                    cat_inputs = torch.cat((id_inputs, ood_inputs), 0)
                    cat_hiddens, cat_outputs = self.forward(cat_inputs)
                    id_hiddens = [cat_hidden[:id_bs] for cat_hidden in cat_hiddens]

                    id_outputs = cat_outputs[:id_bs]
                    ood_outputs = cat_outputs[id_bs:]

                    id_loss = criterion(id_outputs, id_labels)
                    id_conf = F.softmax(id_outputs, dim=1)
                    mean_id_conf = torch.gather(id_conf, dim=1, index=id_labels.unsqueeze(1)).mean()

                    bido_loss = bilateral_dependency_loss(id_inputs, id_hiddens, id_labels,
                                                          num_classes=self.num_classes,
                                                          measure=self.bido_args["params"]["measure"],
                                                          alpha=self.bido_args["params"]["alpha"],
                                                          beta=self.bido_args["params"]["beta"])

                    #########################################################################
                    if self.OE_args['strategy'] == "CM":
                        tmp = F.softmax(ood_outputs, dim=1) * F.log_softmax(ood_outputs, dim=1)
                        ent = -1.0 * tmp.sum()
                        if epoch < ft_epochs:
                            loss = id_loss + bido_loss[0]
                        if epoch >= ft_epochs:
                            loss = id_loss + bido_loss[0] - 0.0001 * ent

                        in_energy_losses.update(ent.data, id_bs)
                    #########################################################################

                    #########################################################################
                    elif self.OE_args['strategy'] == "ER":
                        E = -torch.logsumexp(cat_outputs, dim=1)
                        Ec_in = E[:id_bs]
                        Ec_out = E[id_bs:]

                        if epoch < ft_epochs:
                            in_energy_loss = Ec_in.mean()
                            out_energy_loss = Ec_out.mean()

                            loss = id_loss + bido_loss[0]

                        if epoch < ft_epochs:
                            m_in = in_energy_losses.avg
                            m_out = out_energy_losses.avg

                        if epoch >= ft_epochs:
                            in_energy_loss = torch.pow(F.relu(Ec_in - m_in), 2).mean()
                            out_energy_loss = torch.pow(F.relu(m_out - Ec_out), 2).mean()
                            loss = id_loss + bido_loss[0] + 0.1 * (out_energy_loss + in_energy_loss)

                        in_energy_losses.update(in_energy_loss.data, id_bs)
                        out_energy_losses.update(out_energy_loss.data, ood_bs)
                    #########################################################################

                else:
                    cat_inputs = torch.cat((id_inputs, ood_inputs), 0)
                    cat_outputs = self.forward(cat_inputs)

                    if isinstance(cat_outputs, tuple):
                        cat_outputs = cat_outputs[-1]

                    id_outputs = cat_outputs[:id_bs]
                    ood_outputs = cat_outputs[id_bs:]

                    if epoch < ft_epochs:
                        self.eval()
                        ood_outputs = self.forward(ood_inputs)[-1]
                        self.train()
                        id_outputs = self.forward(id_inputs)[-1]
                    else:
                        cat_inputs = torch.cat((id_inputs, ood_inputs), 0)
                        cat_outputs = self.forward(cat_inputs)

                        if isinstance(cat_outputs, tuple):
                            cat_outputs = cat_outputs[-1]

                        id_outputs = cat_outputs[:id_bs]
                        ood_outputs = cat_outputs[id_bs:]

                    id_loss = criterion(id_outputs, id_labels)
                    id_conf = F.softmax(id_outputs, dim=1)
                    mean_id_conf = torch.gather(id_conf, dim=1, index=id_labels.unsqueeze(1)).mean()

                    #########################################################################
                    if self.OE_args['strategy'] == "CM":
                        tmp = F.softmax(ood_outputs, dim=1) * F.log_softmax(ood_outputs, dim=1)
                        ent = -1.0 * tmp.sum()
                        if epoch < ft_epochs:
                            loss = id_loss
                        if epoch >= ft_epochs:
                            loss = id_loss - 0.001 * ent

                        in_energy_losses.update(ent.data, id_bs)
                    #########################################################################

                    #########################################################################
                    elif self.OE_args['strategy'] == "ER":
                        E = -torch.logsumexp(cat_outputs, dim=1)
                        Ec_in = E[:id_bs]
                        Ec_out = E[id_bs:]

                        if epoch < ft_epochs:
                            in_energy_loss = Ec_in.mean()
                            out_energy_loss = Ec_out.mean()

                            loss = id_loss

                        if epoch < ft_epochs:
                            m_in = in_energy_losses.avg
                            m_out = out_energy_losses.avg

                        if epoch >= ft_epochs:
                            in_energy_loss = torch.pow(F.relu(Ec_in - m_in), 2).mean()
                            out_energy_loss = torch.pow(F.relu(m_out - Ec_out), 2).mean()
                            loss = id_loss + 0.1 * (out_energy_loss + in_energy_loss)

                        in_energy_losses.update(in_energy_loss.data, id_bs)
                        out_energy_losses.update(out_energy_loss.data, ood_bs)
                    #########################################################################

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                running_total_loss += loss * id_bs
                running_id_loss += id_loss * id_bs
                running_bido_loss += bido_loss[0] * id_bs
                running_id_conf += mean_id_conf * id_bs

                metric_train.update(id_outputs, id_labels)

            print(
                f'Training {metric_train.name}:   {metric_train.compute_metric():.2%}',
                f'\t total loss: {running_total_loss / num_training_data:.4f}',
                f'\t id loss: {running_id_loss / num_training_data:.4f}',
                f'\t bido loss: {running_bido_loss / num_training_data:.4f}',
                f'\t id conf: {running_id_conf / num_training_data:.4f}' +
                '\tInE Loss ({in_e_loss.avg:.4f})'
                '\tOutE Loss ({out_e_loss.avg:.4f})'.format(
                    in_e_loss=in_energy_losses,
                    out_e_loss=out_energy_losses)
            )

            # Validation
            if validation_data:
                self.eval()
                val_metric, val_loss, running_id_conf = self.evaluate(
                    validation_data,
                    batch_size,
                    metric,
                    criterion,
                    dataloader_num_workers=dataloader_num_workers)

                ######
                epoch_logger.writerow({
                    'global_iteration': epoch,
                    'train_acc': metric_train.compute_metric(),
                    'test_acc': val_metric,

                })
                plot_csv(epoch_logger.filename, os.path.join(save_base_path, f'epoch_plots.jpeg'))
                ######

                print(
                    f'Validation {metric_train.name}: {val_metric:.2%}',
                    f'\t Validation Loss:  {val_loss:.4f}',
                    f'\t id conf: {running_id_conf / len(validation_data):.4f}',
                )

                # Save best model
                if epoch >= ft_epochs and val_metric > best_model_values['validation_metric']:
                    print('Copying better model')
                    best_model_values['validation_metric'] = val_metric
                    best_model_values['validation_loss'] = val_loss
                    best_model_values['model_state_dict'] = deepcopy(self.state_dict())
                    best_model_values['model_optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                    best_model_values['training_metric'] = metric_train.compute_metric()
                    best_model_values['training_loss'] = running_total_loss / num_training_data

            else:
                best_model_values['validation_metric'] = None
                best_model_values['validation_loss'] = None
                best_model_values['model_state_dict'] = deepcopy(self.state_dict())
                best_model_values['model_optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                best_model_values['training_metric'] = metric_train.compute_metric()
                best_model_values['training_loss'] = running_total_loss / len(trainloader)

            # Update the RTPT
            rtpt.step(subtitle=f"loss={running_total_loss / len(trainloader):.4f}")

            # make the lr scheduler step
            if lr_scheduler is not None:
                lr_scheduler.step()

        if save_base_path:
            if not os.path.exists(save_base_path):
                os.makedirs(save_base_path)
            if validation_data:
                model_path = os.path.join(
                    save_base_path, self.name + f'_{best_model_values["validation_metric"]:.4f}' + '.pth')
            else:
                model_path = os.path.join(
                    save_base_path, self.name + f'_{best_model_values["training_metric"]:.4f}_no_val' + '.pth')

        else:
            model_path = self.name

        torch.save(
            {
                'epoch': num_epochs,
                'model_state_dict': best_model_values['model_state_dict'],
                'optimizer_state_dict': best_model_values['model_optimizer_state_dict'],
            }, model_path)

        val_metric = best_model_values['validation_metric']
        val_loss = best_model_values['validation_loss']
        print(
            f'\nBest Val {metric_train.name}: {val_metric:.2%} \t Val Loss: {val_loss:.4f} \n'
        )
        # Test final model
        test_metric, test_loss = None, None
        if test_data:
            test_metric, test_loss, _ = self.evaluate(
                test_data,
                batch_size,
                metric,
                criterion=nn.CrossEntropyLoss(),
                dataloader_num_workers=dataloader_num_workers)
            print(
                '----------------------- FINISH TRAINING -----------------------'
            )
            print(
                f'Final Test {metric_train.name}: {test_metric:.2%} \t Test Loss: {test_loss:.4f} \n'
            )


    def evaluate(self,
                 evaluation_data,
                 batch_size=128,
                 metric=Accuracy,
                 criterion=nn.CrossEntropyLoss(),
                 dataloader_num_workers=4):
        evalloader = DataLoader(evaluation_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=dataloader_num_workers,
                                pin_memory=True)
        num_val_data = len(evalloader.dataset)
        metric = metric()
        self.eval()

        with torch.no_grad():
            running_id_conf = 0.0
            running_loss = torch.tensor(0.0, device=self.device)
            for inputs, labels in tqdm(evalloader,
                                       desc='Evaluating',
                                       leave=False,
                                       file=sys.stdout):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                model_output = self.forward(inputs)
                if isinstance(model_output, tuple):
                    model_output = model_output[-1]

                id_bs = len(inputs)
                id_conf = F.softmax(model_output, dim=1)
                mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()

                running_id_conf += mean_id_conf * id_bs

                metric.update(model_output, labels)

                running_loss += criterion(model_output,
                                          labels).cpu() * inputs.shape[0]

            metric_result = metric.compute_metric()

            return metric_result, running_loss.item() / num_val_data, running_id_conf

    def dry_evaluate(self,
                     evaluation_data,
                     batch_size=128,
                     metric=Accuracy,
                     criterion=nn.CrossEntropyLoss(),
                     dataloader_num_workers=4,
                     ood_threshold=0.5):
        evalloader = DataLoader(evaluation_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=dataloader_num_workers,
                                pin_memory=True)
        metric = metric()
        self.eval()
        with torch.no_grad():
            running_id_conf = 0.0
            running_ood_conf = 0.0
            running_loss = torch.tensor(0.0, device=self.device)
            for inputs, labels in tqdm(evalloader,
                                       desc='Evaluating',
                                       leave=False,
                                       file=sys.stdout):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                model_output = self.forward(inputs)
                if isinstance(model_output, tuple):
                    model_output = model_output[-1]

                id_conf = F.softmax(model_output, dim=1)
                mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()
                
                id_bs = len(inputs)
                if self.OE_args['enable_OE']:
                    ood_conf = F.softmax(model_output, dim=1)[:, -1].mean()
                    running_ood_conf += ood_conf * id_bs
                
                    if ood_threshold:
                        ### P_k+1 | ood data > 0.5
                        mask = F.softmax(model_output, dim=1)[:, -1] < ood_threshold
                        model_output[:, -1] = model_output[:, -1].masked_fill(mask, -float("inf"))
                
                running_id_conf += mean_id_conf * id_bs

                metric.update(model_output, labels)
                running_loss += criterion(model_output,
                                          labels).cpu() * inputs.shape[0]

            metric_result = metric.compute_metric()

            print(
                f'Validation {metric.name}: {metric_result:.2%}',
                f'\t Validation Loss:  {running_loss.item() / len(evaluation_data):.4f}',
                f'\t id conf: {running_id_conf / len(evaluation_data):.4f}',
                f'\t ood conf: {running_ood_conf / len(evaluation_data):.4f} (OOD > {ood_threshold}) '
            )

        return metric_result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

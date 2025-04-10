import os, time
import torch.optim as optim
import torch
from utils import utils
from models.classifier import Classifier
from metrics.accuracy import Accuracy

def create_optimizer(model, loaded_args):
    optimizer_config = loaded_args['optimizer']
    for optimizer_type, args in optimizer_config.items():
        if not hasattr(optim, optimizer_type):
            raise Exception(
                f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
            )

        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(model.parameters(), **args)
        break

    return optimizer


def create_lr_scheduler(optimizer, loaded_args):
    if not 'lr_scheduler' in loaded_args:
        return None

    scheduler_config = loaded_args['lr_scheduler']
    for scheduler_type, args in scheduler_config.items():
        if not hasattr(optim.lr_scheduler, scheduler_type):
            raise Exception(
                f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
            )

        scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(optimizer, **args)
    return scheduler


def main(loaded_args):
    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    OE_args = loaded_args["outlier_exp"]
    bido_args = loaded_args["bido"]

    print(bido_args)
    print(OE_args)

    model_config = loaded_args["model"]
    target_model = Classifier(**model_config, bido_args=bido_args, OE_args=OE_args)

    # Set up optimizer and scheduler
    optimizer = create_optimizer(target_model, loaded_args)

    lr_scheduler = create_lr_scheduler(optimizer, loaded_args)

    print(OE_args['enable_OE'])
    print(bido_args['enable_bido'])

    # Start training
    if OE_args['enable_OE']:
        target_model.OE_fit(
            train_loader=train_loader,
            val_loader=test_loader,
            test_loader=test_loader,
            aux_ood_set=aux_ood_set,
            criterion=torch.nn.CrossEntropyLoss(reduction='none'),
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=loaded_args["training"]['batch_size'],
            num_epochs=loaded_args["training"]['num_epochs'],
            dataloader_num_workers=loaded_args["training"]['num_workers'],
            save_base_path=save_model_path,
        )
    else:
        target_model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            test_loader=test_loader,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=loaded_args["training"]['batch_size'],
            num_epochs=loaded_args["training"]['num_epochs'],
            dataloader_num_workers=loaded_args["training"]['num_workers'],
            save_base_path=save_model_path
        )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cxr')
    parser.add_argument('--measure', '-m', default='HSIC', help='HSIC | COCO')
    parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')

    parser.add_argument('--enable_bido', action='store_true', help='multi-layer constraints')
    parser.add_argument('--enable_OE', action='store_true', help='OE constraints')
    parser.add_argument('--strategy', '-s', default='CM', help='')

    parser.add_argument('--config', '-c', default='./config/celeba.json', help='')
    parser.add_argument('--save_path', default='./results', help='')

    parser.add_argument('--sampler', action='store_true', help='sampler')
    parser.add_argument('--alpha', '-a', type=float, default=0.05)
    parser.add_argument('--beta', '-b', type=float, default=0.5)

    args = parser.parse_args()

    loaded_args = utils.load_json(json_file=args.config)

    loaded_args['dataset']['sampler'] = args.sampler
    loaded_args['bido']['enable_bido'] = args.enable_bido
    loaded_args['outlier_exp']['enable_OE'] = args.enable_OE
    loaded_args['outlier_exp']['strategy'] = args.strategy

    if not args.enable_bido:
        alpha, beta = loaded_args["bido"]["params"]["alpha"], loaded_args["bido"]["params"]["beta"] = 0, 0
    else:
        loaded_args["bido"]["params"]["measure"] = args.measure
        alpha, beta = loaded_args["bido"]["params"]["alpha"], loaded_args["bido"]["params"]["beta"] = args.alpha, args.beta

    train_file = loaded_args['dataset']['train_file']
    test_file = loaded_args['dataset']['test_file']

    _, train_loader = utils.init_dataloader(loaded_args, train_file, mode="train")
    _, test_loader = utils.init_dataloader(loaded_args, test_file, mode="test")

    if args.enable_OE:
        aux_ood_set, aux_ood_loader = utils.init_dataloader(loaded_args, mode="aux_ood")

    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.enable_OE:
        save_model_path = os.path.join(
            args.save_path,
            f"{args.dataset}_{loaded_args['model']['architecture']}",
            f"{args.measure}_{alpha}_{beta}_OE_{args.strategy}_{loaded_args['outlier_exp']['aux_ood_dataset']}_{time_stamp}"
        )
    else:
        save_model_path = os.path.join(
            args.save_path,
            f"{args.dataset}_{loaded_args['model']['architecture']}",
            f"{args.measure}_{alpha}_{beta}_OE_{args.strategy}_{time_stamp}"
        )
    os.makedirs(save_model_path, exist_ok=True)

    main(loaded_args)

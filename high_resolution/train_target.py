import argparse
import os
import time
import torch

from copy import copy
from metrics.accuracy import Accuracy
from utils.training_config_parser import TrainingConfigParser


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(
        description='Training a target classifier')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')

    parser.add_argument('--enable_bido', action='store_true', help='multi-layer constraints')
    parser.add_argument('--measure', '-m', default='HSIC', help='HSIC | COCO')
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--beta', '-b', type=float, default=0.1)

    parser.add_argument('--enable_OE', action='store_true', help='OE constraints')
    parser.add_argument('--strategy', '-s', default='', help='')

    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load json config file
    config = TrainingConfigParser(args.config.strip())

    config.bido['enable_bido'] = args.enable_bido
    config.outlier_exp['enable_OE'] = args.enable_OE
    config.outlier_exp['strategy'] = args.strategy

    if not args.enable_bido:
        config.bido["params"]["alpha"], config.bido["params"]["beta"] = 0, 0
    else:
        config.bido["params"]["measure"] = args.measure
        config.bido["params"]["alpha"], config.bido["params"]["beta"] = args.alpha, args.beta

    # Set seeds and make deterministic
    seed = config.seed
    torch.manual_seed(seed)

    # Create the target model architecture
    target_model = config.create_model()

    # Build the datasets
    train_set, test_set, aux_ood_set = config.create_datasets()

    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    # Set up optimizer and scheduler
    optimizer = config.create_optimizer(target_model)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # modify the save_path such that subfolders with a timestamp and the name of the run are created
    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.enable_OE:
        save_path = os.path.join(
            config.training['save_path'],
            f"{config.dataset['type']}_{config.model['architecture']}",
            f"{time_stamp}_{args.measure}_{ config.bido['params']['alpha']}_{ config.bido['params']['beta']}_OE_{args.strategy}_{config.outlier_exp['aux_ood_dataset']}"
        )
    else:
        save_path = os.path.join(
            config.training['save_path'],
            f"{config.dataset['type']}_{config.model['architecture']}",
            f"{time_stamp}_{args.measure}_{ config.bido['params']['alpha']}_{ config.bido['params']['beta']}_OE_{args.strategy}_{config.outlier_exp['aux_ood_dataset']}"
        )


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Start training
    if config.outlier_exp['enable_OE']:
        target_model.OE_fit(
            training_data=train_set,
            validation_data=test_set,
            test_data=test_set,
            ood_data=aux_ood_set,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rtpt=rtpt,
            config=config,
            batch_size=config.training['batch_size'],
            num_epochs=config.training['num_epochs'],
            dataloader_num_workers=config.training['dataloader_num_workers'],
            enable_logging=config.wandb['enable_logging'],
            wandb_init_args=config.wandb['args'],
            save_base_path=save_path,
            config_file=args.config
        )
    else:
        target_model.fit(
            training_data=train_set,
            validation_data=test_set,
            test_data=test_set,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rtpt=rtpt,
            config=config,
            batch_size=config.training['batch_size'],
            num_epochs=config.training['num_epochs'],
            dataloader_num_workers=config.training['dataloader_num_workers'],
            enable_logging=config.wandb['enable_logging'],
            wandb_init_args=config.wandb['args'],
            save_base_path=save_path,
            config_file=args.config)


if __name__ == '__main__':
    main()

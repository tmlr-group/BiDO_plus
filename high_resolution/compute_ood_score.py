import argparse
import os

import torch
from utils.attack_config_parser import AttackConfigParser
from utils.ood_score import compute_ood_score, cal_metric, get_and_print_results
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    # Set devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]
    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Load target model and set dataset
    target_model = config.create_target_model()
    target_model_name = target_model.name
    test_set, test_ood_set = config.create_datasets()
    id_acc = target_model.dry_evaluate(test_set)

    target_model.name = target_model_name

    mode = args.mode
    n_classes = config.dataset["num_classes"]
    id_dataset_name = config.dataset["type"].lower()
    ood_dataset_name = config.dataset["ood_type"].lower()

    # ID acc vs. OOD acc
    target_model_config = config.target_model_config

    test_id_loader = DataLoader(test_set,
                                batch_size=64,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    test_ood_loader = DataLoader(test_ood_set,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    print("len ID:", len(test_id_loader.dataset), "len OOD:", len(test_ood_loader.dataset))

    print("OOD:")
    ood_sum_score = compute_ood_score(target_model, test_ood_loader, n_classes, mode)
    print("ID:")
    id_sum_score = compute_ood_score(target_model, test_id_loader, n_classes, mode)
    print()

    auroc, aupr, fpr = get_and_print_results(id_sum_score, ood_sum_score, id_dataset_name, "100")
    results = cal_metric(known=id_sum_score, novel=ood_sum_score, method=mode)

    if not target_model_config['outlier_exp']['enable_OE']:
        ood_acc = 1 - fpr

    sns.set_style("darkgrid")

    fontsize = 20

    if target_model_config['bido']['enable_bido']:
        title = "BiDO Model"
        save_path = f"plot_figures/BiDO_{mode}.pdf"
    else:
        title = "Regular Model"
        save_path = f"plot_figures/Reg_{mode}.pdf"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if mode == 'energy':
        plt.xlabel('Free Energy Score', fontweight="bold", fontsize=fontsize)
    elif mode == 'msp':
        plt.xlabel('MSP Score', fontweight="bold", fontsize=fontsize)

    ax = sns.kdeplot(id_sum_score, label="ID", multiple="stack")
    ax = sns.kdeplot(ood_sum_score, label="OOD", multiple="stack")
    ax.set_title(title, fontweight="bold", fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylabel('Density', fontsize=fontsize)
    plt.savefig(save_path, bbox_inches='tight')


def create_parser():
    parser = argparse.ArgumentParser(
        description='Compute OOD scores!')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    parser.add_argument('--mode',
                        default="msp",
                        type=str,
                        help='OOD score mode: type msp or energy')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


if __name__ == '__main__':
    main()

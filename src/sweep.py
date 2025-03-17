import wandb
import yaml
import subprocess
import argparse
import os


def get_or_create_yaml(sweep_id, base_config_path):
    # Path for a unique YAML file for the sweep
    unique_yaml_path = f"{base_config_path}_{sweep_id}.yaml"

    if not os.path.exists(unique_yaml_path):
        # Assuming base_config_path points to a template YAML file
        with open(base_config_path) as f:
            config_data = yaml.load(f, Loader=yaml.SafeLoader)

        # Save the new YAML file for the sweep
        with open(unique_yaml_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    return unique_yaml_path


def update_yaml_config(file_path, sweep_config):
    """
    Update the YAML configuration file with sweep parameters.
    """
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Update data with sweep_config
    for key, value in sweep_config.items():
        data[key] = value

    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def run_model_with_launcher(yaml_file_path):
    """
    Run the model using the launcher script with the updated YAML configuration.
    """
    command = f"python /home/afc53/contrastive_learning_mri_images/src/launcher.py {yaml_file_path} path=cluster"
    subprocess.run(command, shell=True)


# def parse_args():
#     parser = argparse.ArgumentParser(description="select sweep")

#     parser.add_argument("--sweep", type=str)
#     parser.add_argument("--modality", type=str, default="T1")
#     parser.add_argument("--NN_nb_selection", type=str, default="similarity")

#     return parser.parse_args()


def main():
    base_yaml_file = '/home/afc53/contrastive_learning_mri_images/src/exp/supcon_adam_kernel.yaml'

    with wandb.init() as run:

        sweep_id = run.sweep_id
        config = {k: v for k, v in run.config.items()}

        # config['wandb_name'] = f"{config['NN_nb_selection']}_{config['end_NN_nb']}_{config['NN_nb_step_size']}_" + str(run.id)

        unique_sweep_yaml_path = get_or_create_yaml(sweep_id, base_yaml_file)
        update_yaml_config(unique_sweep_yaml_path, config)

        run_model_with_launcher(unique_sweep_yaml_path)


if __name__ == '__main__':

    # method_config = {'plain_lr_wd_noise': 'random',
    #                  'plain_lr_wd': 'random',
    #                  'plain_epochs': 'grid',
    #                  'plain_wd': 'bayes',
    #                  'plain_noise': 'grid',
    #                  'gamma': 'random',
    #                  'gamma_alignment_repulsion': 'bayes',
    #                  'asym_w': 'bayes',
    #                  'NN_nb': 'random',
    #                  'stratify_bins': 'grid',
    #                  'pair_weight_method': 'grid',
    #                  'mmd': 'bayes',
    #                  'hsic': 'bayes',
    #                  'deep_coral': 'bayes',
    #                  'dynamic_gamma': 'grid',
    #                  'dynamic_gammas': 'grid',
    #                  'dynamic_asym_w': 'grid',
    #                  'dynamic_NN_nbs': 'random',
    #                  'dist_wo_huber_euclidean': 'random',
    #                  'dist_wo_huber_cosine': 'random',
    #                  'dist_wo_huber_manhattan': 'random',
    #                  'dist_wo_huber_chebyshev': 'random',
    #                  'dist_w_huber_euclidean': 'random',
    #                  'dist_w_huber_cosine': 'random',
    #                  'huber_delta': 'random'}

    # count_config = {'plain_lr_wd_noise': 20,
    #                 'plain_lr_wd': 30,
    #                 'plain_epochs': 11,
    #                 'plain_wd': 30,
    #                 'plain_noise': 10,
    #                 'gamma': 20,
    #                 'gamma_alignment_repulsion': 40,
    #                 'asym_w': 40,
    #                 'NN_nb': 15,
    #                 'stratify_bins': 5,
    #                 'pair_weight_method': 3,
    #                 'mmd': 15,
    #                 'hsic': 15,
    #                 'deep_coral': 20,
    #                 'dynamic_gamma': 12,
    #                 'dynamic_gammas': 24,
    #                 'dynamic_asym_w': 24,
    #                 'dynamic_NN_nbs': 1,
    #                 'dist_wo_huber_euclidean': 20,
    #                 'dist_wo_huber_cosine': 20,
    #                 'dist_wo_huber_manhattan': 20,
    #                 'dist_wo_huber_chebyshev': 20,
    #                 'dist_w_huber_euclidean': 20,
    #                 'dist_w_huber_cosine': 20,
    #                 'huber_delta': 10}

    # parameters_config = {
    #     'plain_lr_wd_noise': {
    #         'lr': {
    #             'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    #         },
    #         'weight_decay': {
    #             'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    #         },
    #         'noise_std': {
    #             'values': [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
    #         },
    #         'tf': {
    #             'values': ['noise']
    #         }
    #     },
    #     'plain_lr_wd': {
    #         'lr': {
    #             'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    #         },
    #         'weight_decay': {
    #             'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    #         },
    #     },
    #     'plain_epochs': {
    #         'epochs': {
    #             'values': [2, 3, 5, 10, 20, 30, 40, 50, 60, 75, 100]
    #         }
    #     },
    #     'plain_wd': {
    #         'weight_decay': {
    #             'min': 1e-6,
    #             'max': 1e-3
    #         }
    #     },
    #     'plain_noise': {
    #         'noise_std': {
    #             'min': 0.01,
    #             'max': 0.10
    #         }
    #     },
    #     'gamma': {
    #         'gamma': {
    #             'min': 0.5,
    #             'max': 2.0
    #         }
    #     },
    #     'gamma_alignment_repulsion': {
    #         'gamma_alignment': {
    #             'min': 0.5,
    #             'max': 2.0
    #         },
    #         'gamma_repulsion': {
    #             'min': 0.5,
    #             'max': 2.0
    #         },
    #     },
    #     'asym_w': {
    #         'asym_w_alpha': {
    #             'min': 0.5,
    #             'max': 2.0
    #         },
    #         'asym_w_beta': {
    #             'min': 0.5,
    #             'max': 2.0
    #         },
    #     },
    #     'NN_nb': {
    #         'NN_nb': {
    #             'min': 1,
    #             'max': 32
    #         },
    #     },
    #     'stratify_bins': {
    #         'stratify_bins': {
    #             'values': [6, 8, 9, 10, 12]
    #         }
    #     },
    #     'pair_weight_method': {
    #         'pair_weight_method': {
    #             'values': ['mean', 'min', 'max']
    #         }
    #     },
    #     'mmd': {
    #         'mmd_kernel': {
    #             'values': ['rbf']  # 'rbf', 'linear', 'polynomial'
    #         },
    #         'lambda_site': {
    #             'min': 0.1,
    #             'max': 1.0
    #         },
    #         'lambda_sex': {
    #             'values': [0.0]
    #         },
    #         'lambda_coverage': {
    #             'values': [0.0]
    #         },
    #         'coverage_bin_size': {
    #             'values': [0.1]
    #         },
    #         'epochs': {
    #             'values': [50]
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'hsic': {
    #         'hsic_kernel': {
    #             'values': ['rbf']
    #         },
    #         'lambda_site': {
    #             'min': 0.0,
    #             'max': 1.0
    #         },
    #         'lambda_sex': {
    #             'values': [0.0]
    #         },
    #         'lambda_coverage': {
    #             'values': [0.0]
    #         },
    #         'epochs': {
    #             'values': [50]
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'deep_coral': {
    #         'deep_coral': {
    #             'values': [1]
    #         },
    #         'lambda_site': {
    #             'min': 0.0,
    #             'max': 1.0
    #         },
    #         'lambda_sex': {
    #             'min': 0.0,
    #             'max': 1.0
    #         },
    #         'lambda_coverage': {
    #             'values': [0.0]
    #         },
    #     },
    #     'dynamic_gamma': {
    #         'dynamic_gamma_mode': {
    #             'values': ['Increase', 'Decrease']
    #         },
    #         'dynamic_max_epoch': {
    #             'values': [5, 10, 15, 20, 25, 30]
    #         }
    #     },
    #     'dynamic_gammas': {
    #         'dynamic_gammas_mode': {
    #             'values': ['IncDec', 'DecDec', 'IncInc', 'DecInc']
    #         },
    #         'dynamic_max_epoch': {
    #             'values': [5, 10, 15, 20, 25, 30]
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'dynamic_asym_w': {
    #         'dynamic_asym_w_mode': {
    #             'values': ['IncDec', 'DecDec', 'IncInc', 'DecInc']
    #         },
    #         'dynamic_max_epoch': {
    #             'values': [5, 10, 15, 20, 25, 30]
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'dynamic_NN_nbs': {
    #         #'NN_nb_selection': {
    #         #    'values': ['similarity']
    #         #},
    #         'NN_nb_step_size': {
    #             'values': [1, 2, 5]
    #         },
    #         'end_NN_nb': {
    #             'min': 8,
    #             'max': 14
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'dist_wo_huber_euclidean': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['euclidean']
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'dist_wo_huber_cosine': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['cosine']
    #         },
    #         'pretrained': {
    #             'values': ['expw']
    #         }
    #     },
    #     'dist_wo_huber_manhattan': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['manhattan']
    #         }
    #     },
    #     'dist_wo_huber_chebyshev': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['chebyshev']
    #         }
    #     },
    #     'dist_w_huber_euclidean': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['euclidean']
    #         },
    #         'huber_delta': {
    #             'values': [1.0]
    #         }
    #     },
    #     'dist_w_huber_cosine': {
    #         'dist_alpha': {
    #             'min': 0.0,
    #             'max': 2.0
    #         },
    #         'dist_norm': {
    #             'values': ['cosine']
    #         },
    #         'huber_delta': {
    #             'values': [1.0]
    #         }
    #     },
    #     'huber_delta': {
    #         'huber_delta': {
    #             'min': 0.1,
    #             'max': 1.5
    #         }
    #     }
    # }

    # args = parse_args()

    # modality_parameter_config = {
    #     'modality': {
    #         'values': [args.modality]
    #     },
    # }

    # NN_nb_selection_parameter_config = {
    #     'NN_nb_selection': {
    #         'values': [args.NN_nb_selection]
    #     },
    # }

    # for key in parameters_config:
    #     parameters_config[key].update(modality_parameter_config)

    # for key in parameters_config:
    #     parameters_config[key].update(NN_nb_selection_parameter_config)

    # if args.sweep not in method_config.keys():
    #     raise ValueError('Invalid sweep method')

    sweep_config = {
        'method': 'random',
        "name": "classification_tuning_dynamic_negative_classloss_noGRL_part2",
        'metric': {
            'name': 'train/mae', #'mae_train'
            'goal': 'minimize'
        },
        "parameters": {
        # "batch_size": {"values": [32, 64]},
        # "lr": {"values": [1e-4]},
        "weight_decay": {"values": [1e-6, 1e-2]},
        # # "temp": {"values": [0.05, 0.1, 0.2]},
        # # "method": {"values": ["supcon", "yaware"]},
        # # "optimizer": {"values": ["adam", "sgd"]},
        # # "momentum": {"values": [0, 0.9, 1.0]},
        # # "sigma": {"values": [1, 2]},
        "lambda_adv": {"values": [5e-6, 1e-5, 5e-5, 1e-4]},
        # # "lr_decay_step": {"values": [5, 10, 15]},
        # # "lr_decay_rate": {"values": [0.5, 0.7, 0.9]},
        # # "beta1": {"values": [0.8, 0.9, 0.95]},
        # # "beta2": {"values": [0.99, 0.999, 0.9999]}    
        # # "noise_std": {"values": [0, 0.01, 0.05, 0.1]},
        # # "kernel": {"values": ["gaussian", "rbf"]},
        # # "lr_decay_rate": {"values": [0.1, 0.9]},
        # # "grl_layer": {"values": [1, 0]},
        # "lambda_val": {"values": [0.0005, 0.005, 0.05, 0.5, 5, 50]}
        # "trial": {"values": [0, 1, 2, 3, 4]},

    },
    }

    # sweep_id = wandb.sweep(sweep_config,
    #                        entity='jakobwandb',
    #                        #project='seedsNEW-pretrained-expw-finetune-sweeps-' + args.sweep + '-' + args.modality)
    #                        project='MICCAI_suppl')
    

    sweep_id = wandb.sweep(sweep_config, project="contrastive-brain-age-prediction")



    wandb.agent(sweep_id, function=main, project="contrastive-brain-age-prediction", count=5)

    # wandb.agent(sweep_id,
    #             function=main,
    #             count=count_config[args.sweep])

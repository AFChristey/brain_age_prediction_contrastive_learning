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

    # # Print the updated YAML before saving
    # print("\n🔹 Updated YAML Configuration for Sweep:")
    # print(yaml.dump(data, default_flow_style=False))


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


    sweep_config = {
        'method': 'random',
        # "name": "classification_tuning_dynamic_negative_classloss_noGRL_part2",
        "name": "added_config_test_2",
        'metric': {
            'name': 'train/mae', #'mae_train'
            'goal': 'minimize'
        },
        "parameters": {
        # "batch_size": {"values": [32, 64]},
        # "lr": {"values": [1e-4]},
        # "weight_decay": {"values": [1e-6, 1e-2]},
        # # "temp": {"values": [0.05, 0.1, 0.2]},
        # # "method": {"values": ["supcon", "yaware"]},
        # # "optimizer": {"values": ["adam", "sgd"]},
        # # "momentum": {"values": [0, 0.9, 1.0]},
        # # "sigma": {"values": [1, 2]},
        # "lambda_adv": {"values": [5e-6, 1e-5, 5e-5, 1e-4]},
        # # "lr_decay_step": {"values": [5, 10, 15]},
        # # "lr_decay_rate": {"values": [0.5, 0.7, 0.9]},
        # # "beta1": {"values": [0.8, 0.9, 0.95]},
        # # "beta2": {"values": [0.99, 0.999, 0.9999]}    
        # # "noise_std": {"values": [0, 0.01, 0.05, 0.1]},
        # # "kernel": {"values": ["gaussian", "rbf"]},
        # # "lr_decay_rate": {"values": [0.1, 0.9]},
        # # "grl_layer": {"values": [1, 0]},
        # "lambda_val": {"values": [0.0005, 0.005, 0.05, 0.5, 5, 50]}
        "trial": {"values": [0,1,2]},

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

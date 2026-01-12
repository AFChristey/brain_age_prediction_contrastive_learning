import yaml
import subprocess
import argparse

def create_yaml(base_config_path):


    with open(base_config_path) as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
    
    confound_loss = config_data['confound_loss']
    modality = config_data['modality']
    loss_choice = config_data['loss_choice']


    id = 'NEW' + confound_loss + '_' + modality + '_' + loss_choice

    wandb_name = f"part2_5times_{loss_choice}_{modality}_{confound_loss}"

    # Path for a unique YAML file for the sweep
    unique_yaml_path = f"{base_config_path}_{id}.yaml"


    # Save the new YAML file for the sweep
    with open(unique_yaml_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return unique_yaml_path, wandb_name


def update_yaml_config(file_path, i, wandb_name):
    """
    Update the YAML configuration file with sweep parameters.
    """
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Update data
    data['trial'] = i

    data['wandb_name'] = wandb_name

    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)



def run_model_with_launcher(yaml_file_path):
    """
    Run the model using the launcher script with the updated YAML configuration.
    """
    command = f"python /home/afc53/contrastive_learning_mri_images/src/launcher.py {yaml_file_path} path=cluster"
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    # args = parse_args()
    # id = 'NEW' + args.method + '_' + args.modality + '_' + args.error + '_' + str(args.epochs)
    # for key in args.hpams_dict:
    #     id += f"_{key}={args.hpams_dict[key]}"

    base_yaml_file = '/home/afc53/contrastive_learning_mri_images/src/exp/supcon_adam_kernel.yaml'
    unique_yaml_path, wandb_name = create_yaml(base_yaml_file)

    for i in range(5):
        update_yaml_config(unique_yaml_path, i, wandb_name)
        run_model_with_launcher(unique_yaml_path)











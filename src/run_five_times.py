import yaml
import subprocess
import argparse


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def create_yaml(id, base_config_path):
    # Path for a unique YAML file for the sweep
    unique_yaml_path = f"{base_config_path}_{id}.yaml"

    # Assuming base_config_path points to a template YAML file
    with open(base_config_path) as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)

    # Save the new YAML file for the sweep
    with open(unique_yaml_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return unique_yaml_path


def update_yaml_config(file_path, i, args):
    """
    Update the YAML configuration file with sweep parameters.
    """
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Update data
    if args.error == "trial":
        data['trial'] = i
    elif args.error == "fold":
        data['fold'] = i
    else:
        raise ValueError("Error type not recognized, please use 'trial' or 'fold' as error type.")

    data['epochs'] = args.epochs

    hpams_str = ""
    for key in args.hpams_dict:
        data[key] = args.hpams_dict[key]
        hpams_str += f"_{key}={args.hpams_dict[key]}"
        print(f"Updated {key} to {args.hpams_dict[key]}")

    # data['save_model'] = args.save_model
    data['modality'] = args.modality
    # data['pretrained'] = args.pretrained
    data['method'] = args.method
    #data['wandb_name'] = 'rerun_baseline'
    #data['wandb_name'] = f"NEW-contrastive-pretrain-{args.pretrained}-finetune-{args.method}-E{args.epochs}-{args.modality}-{args.error}-5times"
    data['wandb_name'] = f"NEW-contrastive-pretrain-{args.pretrained}-finetune-{args.method}-{hpams_str}-{args.modality}-{args.error}-5times"

    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def parse_args():
    parser = argparse.ArgumentParser(description="select sweep")

    parser.add_argument("--modality", type=str, default="T1")
    parser.add_argument("--error", type=str, default="trial")
    parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--pretrained", type=str, default="no")
    # parser.add_argument("--save_model", type=int, default=0)
    # parser.add_argument("--hpams_dict", dest='hpams_dict', action=StoreDictKeyPair, default={})
    parser.add_argument("--method", type=str, choices=["threshold", "expw", "yaware"], default="expw")

    return parser.parse_args()


def run_model_with_launcher(yaml_file_path):
    """
    Run the model using the launcher script with the updated YAML configuration.
    """
    command = f"python /home/afc53/contrastive_learning_mri_images/src/launcher.py {yaml_file_path} path=cluster"
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    args = parse_args()
    id = 'NEW' + args.method + '_' + args.modality + '_' + args.error + '_' + str(args.epochs)
    # for key in args.hpams_dict:
    #     id += f"_{key}={args.hpams_dict[key]}"

    base_yaml_file = '/home/afc53/contrastive_learning_mri_images/src/exp/supcon_adam_kernel.yaml'
    unique_yaml_path = create_yaml(id, base_yaml_file)

    for i in range(5):
        update_yaml_config(unique_yaml_path, i, args)
        run_model_with_launcher(unique_yaml_path)

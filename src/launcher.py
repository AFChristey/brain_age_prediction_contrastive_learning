import yaml
import sys
import subprocess
import os
from yaml.loader import SafeLoader


if __name__ == '__main__':
    # If no YAML file path is provided (len(sys.argv) <= 1), 
    # the script prints a usage message and exits with a status code 1 (indicating an error)
    if len(sys.argv) <= 1:
        print("Usage: ./launcher.py path/to/yaml")
        exit(1)

    # This opens the YAML file specified as the first command-line argument (sys.argv[1]).
    # It uses the yaml.load() function to parse the YAML file into a Python dictionary (data). 
    # The SafeLoader is used to ensure safe loading (i.e., without executing any arbitrary code embedded in the YAML).
    with open(sys.argv[1]) as f:
        data = yaml.load(f, Loader=SafeLoader)

    # print(sys.argv[3])

    if len(sys.argv) > 3 and sys.argv[3] == "testing":
        program = "confound_metrics.py"
    else:
        # Recieves the program from yaml file - in this case 'main_mse.py'
        program = data['program']


    # removes the 'program' entry from the data dictionary since it is no longer needed, 
    # leaving the remaining configuration parameters for command-line arguments.
    del data['program']

    skip = False

    # allows the user to override values from the YAML file via command-line arguments
    # sys.argv[2:] represents all command-line arguments after the script name and YAML file path.
    for idx, override in enumerate(sys.argv[2:3]):
        if skip: 
            skip = False
            continue
        
        if '=' in override:
            k, v = override.split('=')
        else:
            k = override.replace('--', '')
            v = sys.argv[2+idx+1]
            skip = True
        data[k] = v

    # print(os.path.join(os.getcwd(), program))
    if data["path"] == "cluster":
        program = "/home/afc53/contrastive_learning_mri_images/src/" + program

    # print(os.path.join(os.getcwd(), program))
        

    # builds the command to run the Python program
        # 'os.getcwd()' returns the current working directory, and 'program' is the path to the script specified in the YAML file.
    args = ["python3", os.path.join(os.getcwd(), program)]

    # The for loop iterates over the data dictionary and appends each key-value pair as a command-line argument in the format --key value
    for k, v in data.items():
        args.extend(["--" + k, str(v)])
    # Prints the full command that will be executed, which includes the Python command, program path, and all arguments
    print("Running:", ' '.join(args))
    # Executes the command using subprocess.run(), which runs the command as a new process. 
    # This will start the Python script specified in the YAML with the arguments from data
    subprocess.run(args)

"""
This file will select the questions used in current experiment iteration
"""

##### Libraries
import json, random, yaml, argparse
from pathlib import Path
print(Path.cwd())
from src.data.hotpot_loader import load_hotpotqa


def main():
    #### Create argument parser
    ### Initialized
    parser = argparse.ArgumentParser(
        description='YAML filename, which will be same to dump and freeze experiment questions'
    )
    ### File path
    parser.add_argument(
        'yaml_filename',
        type=str,
        help='YAML filename'
    )
    ### Verbosity
    parser.add_argument(
        '--v',
        action='store_true',
        help='Enable verbose output'
    )
    ### Parse arguments
    args = parser.parse_args()


    #### Extracting YAML info
    ### Verifying yaml file
    yaml_path = f'{Path.cwd()}/configs/{args.yaml_filename}.yaml'
    if not Path(yaml_path).exists():
        raise FileNotFoundError('File not found, put a new filename')
    ### Loading YAML
    config = yaml.safe_load(
        Path(yaml_path).read_text()
    )
    ### Extracting info
    seed = config['experiment']['seed']
    num_questions = config['dataset']['n_questions']


    #### Extracting questions
    ### Loading HotpotQA
    questions = load_hotpotqa()



    if args.v:
        print(questions)





if __name__ == "__main__":
    main()
"""
This file will select the questions used in current experiment iteration
"""

##### Libraries
import json, random, yaml, argparse
from pathlib import Path
from src.data.hotpot_loader import load_hotpotqa

##### Hyperparam
CWD = Path.cwd()

##### Main
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
    yaml_path = CWD.joinpath(
        'configs', 
        f'{args.yaml_filename}.yaml'
    )
    if not Path(yaml_path).exists():
        raise FileNotFoundError('File not found, put a new filename.')
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
    ### Random generator
    rand_gen = random.Random(seed)
    ### Shuffling questions
    rand_gen.shuffle(questions)
    ### Selecting questions
    selected = questions[:num_questions]


    #### Saving Questions
    ### Output path
    save_output = CWD.joinpath(
        'data', 
        'processed',
        f'{args.yaml_filename}_questions.jsonl'
    )
    ### Edge case
    if Path(save_output).exists():
        raise FileExistsError('File already exists, choose different name.')
    ### Path creation
    save_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    ### Saving
    questions_ids = []
    with save_output.open('w') as f:
        for q in selected:
            questions_ids.append(q.question_id)
            f.write(
                json.dumps(
                    q.__dict__,
                    default=lambda x: x.__dict__,
                )
                + "\n"
            )


    #### Freezing Question IDs
    ### Output path
    freeze_output = CWD.joinpath(
        'data', 
        'processed',
        f'{args.yaml_filename}_question_ids.json'
    )
    ### Edge case
    if Path(freeze_output).exists():
        raise FileExistsError('File already exists, choose different name.')
    ### Path creation
    freeze_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    ### Freezing questions
    frozen = {
        'seed': seed,
        'dataset': 'hotpotqa',
        'split': 'distractor',
        'n_questions': num_questions,
        'question_ids': questions_ids
    }
    with freeze_output.open('w') as f:
        json.dump(frozen, f)



    if args.v:
        print(f'Saved {len(selected)} to {save_output}')




if __name__ == "__main__":
    main()
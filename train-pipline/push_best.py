import wandb
import csv
import glob
import os
import click


def log_to_w_b(file, mtype, project='mt-t5-ria-news', artifact_name='mT5_model_tite', metadata={}):
    run = wandb.init(project=project,
                     tags=['mT5', 'titles', mtype, 'save_model'], job_type='save_model')
    artifact = wandb.Artifact(artifact_name, type='model',
                              metadata=metadata)
    artifact.add_file(file)
    run.log_artifact(artifact)
    run.finish()


def get_score_from_data(path):
    with open(path, 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        next(reader)
        for it in reader:
            if it[2] == 'mean score':
                return it[3]


def push_best_model(path_to_generated_dataset_with_score, models_path, wand_projekt, dataset_len):
    score_name_dict = {}
    path_to_pt = models_path + '/'

    for data_path in glob.glob(path_to_generated_dataset_with_score + '/*.tsv'):
        name = os.path.basename(data_path)[:-4]
        score_name_dict[name] = get_score_from_data(data_path)

    max_value = max(score_name_dict.values())
    max_keys = [k for k, v in score_name_dict.items() if v == max_value]
    log_to_w_b(path_to_pt + max_keys[0] + '.pt', 'google/mt5-small', project=wand_projekt,
               artifact_name='T5_model_tasks',
               metadata={"cos_score": max_value, "train_dataset_len": dataset_len})


@click.command()
@click.option('--path_to_generated_dataset_with_score', type=click.Path(exists=True), help='Path to dataset with score')
@click.option('--models_path', type=click.Path(exists=True), help='Path to models directory')
@click.option('--wand_projekt', default='mt-t5-ria-news')
@click.option('--dataset_len', default=100000)
def run_push_best_model(path_to_generated_dataset_with_score, models_path, wand_projekt, dataset_len):
    push_best_model(path_to_generated_dataset_with_score, models_path, wand_projekt, dataset_len)

if __name__ == '__main__':
    run_push_best_model()

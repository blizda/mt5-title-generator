import csv
import glob
import os
from simple_elmo import ElmoModel
from sklearn.metrics.pairwise import cosine_similarity
import statistics
import click
import wandb


def get_model_from_w_b_without_run(artifact_name='elmo_model:latest'):
    api = wandb.Api()
    artifact = api.artifact('blizd/mt-t5-ria-news/'+artifact_name, type='model')
    return artifact.download()


def generate_eval_data(result_path, text, gt_rs, generated, score):
    with open(result_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['text', 'GT result', 'model result', 'cos score'])
        all_simularity = []
        for i in range(len(text)):
            result_list = []
            result_list.append(text[i])
            result_list.append(gt_rs[i])
            result_list.append(generated[i])
            result_list.append(score[i][i])
            all_simularity.append(score[i][i])
            writer.writerow(result_list)
        writer.writerow([' ', ' ', 'mean score', statistics.mean(all_simularity)])


def iterate_over_dataset(path, model, output):
    with open(path, 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        next(reader)
        temp_dat_text = []
        temp_dat_gt = []
        temp_dat_gen = []
        for i, it in enumerate(reader):
            temp_dat_text.append(it[0])
            temp_dat_gt.append(it[1])
            temp_dat_gen.append(it[2])
        vectors_gt = model.get_elmo_vector_average(temp_dat_gt)
        vectors_gen = model.get_elmo_vector_average(temp_dat_gen)
        sim = cosine_similarity(vectors_gt, vectors_gen)
        generate_eval_data(output, temp_dat_text, temp_dat_gt, temp_dat_gen, sim)


def generate_score(generated_dataset_path, dataset_to_save_with_score):
    path = get_model_from_w_b_without_run()
    model = ElmoModel()
    model.load(path, max_batch_size=32)
    for data_path in glob.glob(generated_dataset_path +'/*.tsv'):
        name = os.path.basename(data_path)
        iterate_over_dataset(data_path, model, dataset_to_save_with_score + '/' + name)

@click.command()
@click.option('--generated_dataset_path', type=click.Path(exists=True), help='Path to generated dataset')
@click.option('--dataset_to_save_with_score', type=click.Path(exists=True), help='Path to directory for '
                                                                                 'saving dataset with score')
def run_generate_score(generated_dataset_path, dataset_to_save_with_score):
    generate_score(generated_dataset_path, dataset_to_save_with_score)


if __name__ == '__main__':
    run_generate_score()

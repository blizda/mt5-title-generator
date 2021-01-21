import json
import re
import csv
from transformers import T5Tokenizer
import click
import wandb
import glob
import os

def get_dataset_from_w_b_without_run(artifact_name='ria_dataset:latest'):
    api = wandb.Api()
    artifact = api.artifact('blizd/mt-t5-ria-news/'+artifact_name, type='dataset')
    return artifact.download()

def cleaning_text(str_in):
    new_string = re.sub('<\/?\w+\W?\/?>', '', str_in)
    new_string = re.sub('<a.+\/?>', '', new_string)
    new_string = re.sub('&\w+;', '', new_string)
    new_string = re.sub('<p.+\/?>', '', new_string)
    new_string = re.sub('<td.+\/?>', '', new_string)
    new_string = re.sub('<span.+\/?>', '', new_string)
    new_string = re.sub('<v.+\/?>', '', new_string)
    new_string = re.sub('<img.+\/?>', '', new_string)
    new_string = re.sub('\n+', ' ', new_string)
    return new_string


def write_to_dataset(json_string, writer, tokenizer, k, max_model_len=1024):
    data_json = json.loads(json_string)
    title = data_json['title']
    new_string = cleaning_text(data_json['text'])
    tknzd = tokenizer.tokenize('напиши заголовок: ' + new_string)
    if len(tknzd) < max_model_len and len(tknzd) > 20:
        writer.writerow([new_string, title])
        k += 1
    return k


def prepare_dataset(train_dataset_save_path, val_dataset_save_path, model_type, train_dataset_len,
                    val_dataset_len, max_model_len):
    dataset = get_dataset_from_w_b_without_run()
    dataset = glob.glob(os.path.join(dataset, '*.json'))[0]
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    k = 0
    with open(dataset) as file, open(
            train_dataset_save_path, 'w') as file_train, open(
            val_dataset_save_path, 'w') as file_val:
        tsv_train = csv.writer(file_train, delimiter='\t')
        tsv_val = csv.writer(file_val, delimiter='\t')
        for it in file:
            if k < train_dataset_len:
                k = write_to_dataset(it, tsv_train, tokenizer, k, max_model_len=max_model_len)
            elif k >= train_dataset_len and k < train_dataset_len + val_dataset_len:
                k = write_to_dataset(it, tsv_val, tokenizer, k, max_model_len=max_model_len)
            else:
                break

@click.command()
@click.option('--train_dataset_save_path', type=click.Path(exists=True), help='Path to save train dataset')
@click.option('--val_dataset_save_path', type=click.Path(exists=True), help='Path save to val dataset')
@click.option('--model_type', default='google/mt5-small')
@click.option('--train_dataset_len', default=100000)
@click.option('--val_dataset_len', default=2000)
@click.option('--max_model_len', default=1024)
def run_prepare(train_dataset_save_path, val_dataset_save_path, model_type, train_dataset_len,
                    val_dataset_len, max_model_len):
    prepare_dataset(train_dataset_save_path, val_dataset_save_path, model_type, train_dataset_len,
                    val_dataset_len, max_model_len)

if __name__ == '__main__':
    run_prepare()
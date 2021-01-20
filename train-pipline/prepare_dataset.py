import json
import re
import csv
from transformers import T5Tokenizer
import click


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

@click.command()
@click.option('--train_dataset', type=click.Path(exists=True), help='Path to train dataset')
@click.option('--val_dataset', type=click.Path(exists=True), help='Path to val dataset')
@click.option('--model_type', default='google/mt5-small')
@click.option('--epochs', default=10)
@click.option('--batch_size', default=1)
@click.option('--wand_projekt', default='mt-t5-ria-news')
@click.option('--checkpoint_path', type=click.Path(exists=True), default='mt5_chkpnt')
@click.option('--lr', default=1e-4)
@click.option('--gpus', default=-1)
@click.option('--precision', default=32)
@click.option('--grad_accum_steps', default=32)
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
train_len = 100000
val_len = 2000
max_model_len = 1024
k = 0
with open('/content/drive/My Drive/ria_dataset/processed-ria.json') as file, open('/content/drive/My Drive/ria_dataset/ria_train_100k.tsv', 'w') as file_train, open('/content/drive/My Drive/ria_dataset/ria_val_2k.tsv', 'w') as file_val:
    tsv_train = csv.writer(file_train, delimiter='\t')
    tsv_val = csv.writer(file_val, delimiter='\t')
    for it in file:
      if k < train_len:
        k = write_to_dataset(it, tsv_train, tokenizer, k, max_model_len=max_model_len)
      elif k >= train_len and k < train_len + val_len:
        k = write_to_dataset(it, tsv_train, tokenizer, k, max_model_len=max_model_len)
      else:
        break
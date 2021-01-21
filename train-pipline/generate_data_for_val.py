from transformers import MT5ForConditionalGeneration, T5Tokenizer
import csv
import glob
import os
import torch
import click


def load_model(model_path, device='cuda'):
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model


def load_dataset(dataset_path):
    dataset = []
    gt_rs = []
    with open(dataset_path, 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        for it in reader:
            dataset.append('напиши заголовок: ' + it[0])
            gt_rs.append(it[1])
    return dataset, gt_rs


def generate_result(dataset, model, tokenizer, device='cpu', beam=2):
    result_dataset = []
    for it in dataset:
        tokenized_text = tokenizer.encode(it, truncation=True, max_length=1024, return_tensors="pt")
        with torch.no_grad():
            tokenized_text = tokenized_text.to(device)
            summary_ids = model.generate(tokenized_text,
                                            num_beams=beam,
                                            no_repeat_ngram_size=2,
                                            min_length=3,
                                            max_length=100,
                                            early_stopping=True
                                         )
            tasks = summary_ids.tolist()
            result_dataset.append(tokenizer.decode(tasks[0], skip_special_tokens=True))
    return result_dataset


def generate_eval_data(result_path, dataset, gt_rs, result_dataset):
    with open(result_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['text', 'GT result', 'model result'])
        for i in range(len(result_dataset)):
            result_list = []
            result_list.append(dataset[i])
            result_list.append(gt_rs[i])
            result_list.append(result_dataset[i])
            writer.writerow(result_list)


def generate_data(val_dataset, models_path, save_generate_file_path, model_type, beam, device):
    tokenizer = T5Tokenizer.from_pretrained(model_type)
    dataset_l, gt_r = load_dataset(val_dataset)
    for vanila_path in glob.glob(models_path + '/*.pt'):
        name = os.path.basename(vanila_path)[:7] + '.tsv'
        model = load_model(vanila_path, device=device)
        result_dataset = generate_result(dataset_l, model, tokenizer, device=device, beam=beam)
        generate_eval_data(save_generate_file_path + '/' + name, dataset_l, gt_r,
                           result_dataset)


@click.command()
@click.option('--val_dataset', type=click.Path(exists=True), help='Path to val dataset')
@click.option('--models_path', type=click.Path(exists=True), help='Path to models directory')
@click.option('--save_generate_file_path', type=click.Path(exists=True), help='Path to generated files directory')
@click.option('--model_type', default='google/mt5-small')
@click.option('--beam', default=2)
@click.option('--device', default='cpu')
def run_generate_data(val_dataset, models_path, save_generate_file_path, model_type, beam, device):
    generate_data(val_dataset, models_path, save_generate_file_path, model_type, beam, device)

if __name__ == '__main__':
    run_generate_data()
import click
from customise_click import CommandConfigFile
from train import train
from prepare_dataset import prepare_dataset
from convert_models import convert
from generate_data_for_val import generate_data
from calc_score_on_val_data import generate_score
from push_best import push_best_model

@click.command(cls=CommandConfigFile('config_file'))
@click.option("--config_file", type=click.Path())
@click.option("--prepare_data")
@click.option("--train")
@click.option("--convert_models")
@click.option("--generate_data_for_val")
@click.option("--calc_score_on_val_data")
@click.option("--push_best")
@click.option('--train_dataset_len')
@click.option('--val_dataset_len')
@click.option('--max_model_len')
@click.option('--train_dataset', type=click.Path(exists=True), help='Path to train dataset')
@click.option('--val_dataset', type=click.Path(exists=True), help='Path to val dataset')
@click.option('--model_type')
@click.option('--epochs')
@click.option('--batch_size')
@click.option('--wand_projekt')
@click.option('--checkpoint_path', type=click.Path(exists=True))
@click.option('--lr')
@click.option('--gpus')
@click.option('--precision')
@click.option('--grad_accum_steps')
@click.option('--save_orig_torch_path', type=click.Path(exists=True))
@click.option('--save_generate_file_path', type=click.Path(exists=True), help='Path to generated files directory')
@click.option('--beam')
@click.option('--device')
@click.option('--dataset_to_save_with_score', type=click.Path(exists=True), help='Path to directory for '
                                                                                 'saving dataset with score')
def run_pipline(config_file, prepare_data, train, convert_models, generate_data_for_val, calc_score_on_val_data,
                push_best, train_dataset_len, val_dataset_len, max_model_len,
                train_dataset, val_dataset, model_type, epochs, batch_size, wand_projekt, checkpoint_path,
                lr, gpus, precision, grad_accum_steps, save_orig_torch_path, save_generate_file_path,
                beam, device, dataset_to_save_with_score):
    if prepare_data:
        prepare_dataset(train_dataset, val_dataset, model_type,
                        train_dataset_len, val_dataset_len, max_model_len)
    if train:
        train(train_dataset, val_dataset, model_type, epochs, batch_size, wand_projekt,
              checkpoint_path, lr, gpus, precision, grad_accum_steps)
    if convert_models:
        convert(checkpoint_path, save_orig_torch_path)
    if generate_data_for_val:
        generate_data(val_dataset, save_orig_torch_path, save_generate_file_path, model_type, beam, device)
    if calc_score_on_val_data:
        generate_score(save_generate_file_path, dataset_to_save_with_score)
    if push_best:
        push_best_model(dataset_to_save_with_score, save_orig_torch_path, wand_projekt, train_dataset_len)

if __name__ == '__main__':
    run_pipline()
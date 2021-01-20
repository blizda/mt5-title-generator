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
@click.option('--save_orig_torch_path', type=click.Path(exists=True), default='mt5_chkpnt_orig_torch')
@click.option('--save_generate_file_path', type=click.Path(exists=True), help='Path to generated files directory')
@click.option('--beam', default=2)
@click.option('--device', default='cpu')
@click.option('--dataset_to_save_with_score', type=click.Path(exists=True), help='Path to directory for '
                                                                                 'saving dataset with score')
@click.option('--elmo_path', type=click.Path(exists=True), help='Path to generated files directory')
def run_pipline(config_file, prepare_data, train, convert_models, generate_data_for_val, calc_score_on_val_data,
                push_best, train_dataset, val_dataset, model_type, epochs, batch_size, wand_projekt, checkpoint_path,
                lr, gpus, precision, grad_accum_steps, save_orig_torch_path, beam, device, dataset_to_save_with_score):
    print(config_file)
    print(prepare_data)

if __name__ == '__main__':
    run_pipline()
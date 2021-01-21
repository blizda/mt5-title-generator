import glob
import os
import click
import torch
from wrapping_mt5 import LtT5model


def convert(checkpoint_path, save_orig_torch_path):
    lt_model = LtT5model()
    for lt_file in glob.glob(checkpoint_path + '/*.ckpt'):
        name = os.path.basename(lt_file)[:7] + '.pt'
        model = lt_model.load_from_checkpoint(lt_file)
        torch.save(model.model.state_dict(), save_orig_torch_path + name)


@click.command()
@click.option('--checkpoint_path', type=click.Path(exists=True), default='mt5_chkpnt')
@click.option('--save_orig_torch_path', type=click.Path(exists=True), default='mt5_chkpnt_orig_torch')
def run_convert(checkpoint_path, save_orig_torch_path):
    convert(checkpoint_path, save_orig_torch_path)


if __name__ == '__main__':
    run_convert()

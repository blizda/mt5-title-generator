from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from wrapping_mt5 import LtT5model
from dataloader import TSVDataset
from torch.utils.data import DataLoader
import click


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
def train(train_dataset, val_dataset, model_type, epochs, batch_size, wand_projekt, checkpoint_path, lr, gpus, precision, grad_accum_steps):
    train_dataset = TSVDataset(train_dataset)
    val_dataset = TSVDataset(val_dataset)
    wandb_logger = WandbLogger(project=wand_projekt, tags=['mT5', 'titles'])
    lt_model = LtT5model(mtype=model_type, lr=lr)
    checkpoint_callback = ModelCheckpoint(monitor='avg_val_loss', filepath=checkpoint_path + '/',
                                          save_top_k=epochs)
    trainer = Trainer(logger=wandb_logger, max_epochs=epochs,
                      checkpoint_callback=checkpoint_callback,
                      gpus=gpus, precision=precision, log_every_n_steps=10, accumulate_grad_batches=grad_accum_steps,
                      flush_logs_every_n_steps=50)
    trainer.fit(lt_model, train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                val_dataloaders=DataLoader(val_dataset, batch_size=batch_size))


if __name__ == '__main__':
    train()

import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))

    early_stop_callback = pl.callbacks.EarlyStopping(patience=20)
    path_format = "{epoch}-{val_loss:.2f}-{%s:.3f}" % cfg.metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=cfg.metric,
        mode="max",
        save_top_k=2,
        filepath=os.path.join(
            cfg.task_model.name, path_format
        ),
        verbose=True,
    )

    tb_logger = TensorBoardLogger("logs", cfg.task_model.name)
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
        logger=tb_logger,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()

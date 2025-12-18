import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback


class SetupCallback(Callback):
    def __init__(self,  now, logdir, ckptdir, cfgdir, config, argv_content=None):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.argv_content = argv_content

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(self.config,
                        os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        
        with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
            f.write(str(self.argv_content))
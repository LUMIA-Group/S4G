import wandb
import glob
import time
from utils import get_timestamp


if __name__ == '__main__':
    # source code and files saving
    wandb.init(name='%s_backup'%(get_timestamp()), notes='Backup source code.', reinit=True, save_code=True, project='S4GNN')
    artifact = wandb.Artifact('source_code', type='code')

    for filename in glob.glob('**/*.py', recursive=True):
        shortname = filename.split('/')[-1]
        if ('test' not in shortname) and ('wandb' not in filename) and ('__init__.py' != shortname):
            new_filename = filename.replace('/', '-')
            artifact.add_file(filename, name=new_filename)

    wandb.log_artifact(artifact)
    wandb.finish()

    time.sleep(3)
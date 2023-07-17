from src.methods.dirty.dirty import DirtyMethod
from src.utils.utils import common_load_dataset


if __name__ == '__main__':
    train_set, dev_set, test_set = common_load_dataset('dirty')
    config = {
        'cuda': False,
        'percent': 0.1,
        'exp_name': 'dirty',
    }
    dirty = DirtyMethod(train_set, dev_set, test_set, config, 'methods/dirty/configs/multitask.xfmr.jsonnet',ckpt = 'methods/dirty/data/dirty_mt.ckpt')
    dirty.evaluate()

from oklac.component import Labeler,RelationExtractor
from oklac.config.new_config import configs as config
import argparse

if __name__ == '__main__':
    tokenizer = Labeler(config)
    tokenizer.train()
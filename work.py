from oklac.component import Labeler,RelationExtractor
from oklac.config.new_config import configs as config
import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0) 

args = parser.parse_args()
if __name__ == '__main__':
    tokenizer = RelationExtractor(config, args.local_rank)
    tokenizer.train()
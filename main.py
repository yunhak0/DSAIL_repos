from src.argument import parse_args
from src.utils import set_random_seeds
import torch

def main():
    args = parse_args()
    set_random_seeds(0)
    torch.set_num_threads(2)

    if args.embedder == 'our_model':
        from models import our_trainer
        embedder = our_trainer(args)

    embedder.train()

if __name__ == '__main__':
    main()

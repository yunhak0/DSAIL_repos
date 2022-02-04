import os
import torch
import src.data import Dataset

class embedder:
    def __init__(self, args):
        self.args = args

        # GPU Setting
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(self.device)

        # Dataset
        self.data = Dataset(root=args.root, dataset=args.dataset)

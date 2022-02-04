import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='data')
    parser.add_argument('--dataset', '-d', type=str, default='cora',
                        help='Dataset names')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--decay', type=float, default=1e-05, help='Weighted Decay')

    parser.add_argument('--device', type=int, default=3, help="GPU to use")

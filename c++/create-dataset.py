import argparse
import pandas as pd
from sklearn import datasets



parser = argparse.ArgumentParser(description='Create Dataset.')
parser.add_argument('n', default=1000, metavar='N', type=int,
                    help='num examples')
parser.add_argument('m', default=200, metavar='M', type=int,
                    help='num fetures')
args = parser.parse_args()
x,y = datasets.make_classification(n_samples=args.n, n_features=args.m)
y[y==0]=-1
pd.DataFrame(x).to_csv('data/{}_{}_data.csv'.format(args.n, args.m),index=False,header=False)
pd.DataFrame(y).to_csv('data/{}_{}_label.csv'.format(args.n, args.m),index=False,header=False)


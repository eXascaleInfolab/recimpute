import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')
from recimpute import main as recimpute_main
from tqdm import tqdm

def main():

    fes_configs = [
        'TSFresh,Topological,Catch22',
        'TSFresh,Topological',
        'TSFresh,Catch22',
        'Topological,Catch22',
        'TSFresh',
        'Topological',
        'Catch22',
    ]

    for fes in tqdm(fes_configs):
        res = recimpute_main(['recimpute.py', '-mode', 'train', '-fes', fes, '-train_for_production', 'False'])
        recimpute_main(['recimpute.py', '-mode', 'eval', '-id', str(res[0].id)])


if __name__ == '__main__':
    main()
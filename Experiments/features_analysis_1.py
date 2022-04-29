import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')
from recimpute import main as recimpute_main

def main():

    fes_configs = [
        'TSFresh,Topological,Catch22,Kats',
        'TSFresh,Topological,Catch22',
        'TSFresh,Topological,Kats',
        'TSFresh,Catch22,Kats',
        'Topological,Catch22,Kats',
        'TSFresh,Topological',
        'TSFresh,Catch22',
        'TSFresh,Kats',
        'Topological,Catch22',
        'Topological,Kats',
        'Catch22,Kats',
        'TSFresh',
        'Topological',
        'Catch22',
        'Kats',
    ]

    for fes in fes_configs:
        res = recimpute_main(['recimpute.py', '-mode', 'train', '-fes', fes, '-train_on_all_data', 'False'])
        recimpute_main(['recimpute.py', '-mode', 'eval', '-id', str(res[0].id)])


if __name__ == '__main__':
    main()
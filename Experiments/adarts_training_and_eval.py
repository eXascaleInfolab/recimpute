import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')
from recimpute import main as recimpute_main

def main():

    res = recimpute_main(['recimpute.py', '-mode', 'train', '-fes', 'TSFresh,Topological,Catch22', '-train_on_all_data', 'False'])
    recimpute_main(['recimpute.py', '-mode', 'eval', '-id', str(res[0].id)])


if __name__ == '__main__':
    main()
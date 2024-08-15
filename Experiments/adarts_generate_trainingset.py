import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')
from recimpute import main as recimpute_main

def main():

    recimpute_main(['recimpute.py', '-mode', 'cluster'])
    recimpute_main(['recimpute.py', '-mode', 'label'])
    recimpute_main(['recimpute.py', '-mode', 'extract_features', 'TSFresh,Topological,Catch22'])


if __name__ == '__main__':
    main()
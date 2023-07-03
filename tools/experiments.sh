#!/bin/bash

# ./model-checker.py modelPy/single-transition.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --note ordinal
# ./model-checker.py modelPy/single-transition.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --onehot-all --note onehot

# ./model-checker.py modelPy/success-fail.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --note ordinal
# ./model-checker.py modelPy/success-fail.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --onehot-all --note onehot

# ./model-checker.py modelPy/safe-risk.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --note ordinal
# ./model-checker.py modelPy/safe-risk.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R1 --onehot-all --note onehot
# ./model-checker.py modelPy/safe-risk.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R2 --note ordinal
# ./model-checker.py modelPy/safe-risk.py --deep-q-learning -v --plot --alpha 1e-4 -k 1000 --properties R2 --onehot-all --note onehot

./model-checker.py modelPy/eajs.py --deep-q-learning -v --plot --alpha 1e-4 -k 500 --properties ExpUtil --onehot-all --batch-size 512 --note onehot-batch512
./model-checker.py modelPy/eajs.py --deep-q-learning -v --plot --alpha 1e-4 -k 500 --properties ExpUtil --onehot-all --batch-size 512 --double-q --note onehot-double-batch512

exit 0
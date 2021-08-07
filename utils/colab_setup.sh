#!bin/bash
cd .. && git checkout causal_mnist
pip3 install argparse_dataclass
python -m utils.mnist


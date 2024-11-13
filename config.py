# configs for CIFAR10

NUM_CLASS = 10
NUM_TRAIN = 50000
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128
SUBSET    = 10000   #25000
ADDENDUM  = 1000  #2500


TRIALS = 5
CYCLES = 10 #7

EPOCH = 200  #200
LR = 0.1
MILESTONES = [160]
EPOCHL = 120

MOMENTUM = 0.9
WDECAY = 5e-4


SCHEME = 1  # 0: expected-gradnorm scheme;   1: entropy-grandorm scheme

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambd

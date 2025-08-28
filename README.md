# kenlm

### Running
```sh
usage: main.py [-h] [-n N] [--test_len TEST_LEN] [-u UNKNOWN_THRESHOLD] [-o OUTPUT] [-p PRUNE] input train_len

Train, validate, and serialize an n-gram model

positional arguments:
  input                 Input training and validation file
  train_len             Length of training set

options:
  -h, --help            show this help message and exit
  -n N                  n-gram model depth
  --test_len TEST_LEN   Length of testing set
  -u UNKNOWN_THRESHOLD, --unknown_threshold UNKNOWN_THRESHOLD
                        Threshold for including words
  -o OUTPUT, --output OUTPUT
                        Binary output file
  -p PRUNE, --prune PRUNE
                        Prune the model
```

ex.
```sh
python3 main.py --test-len=100000 en.tok 3000000
```

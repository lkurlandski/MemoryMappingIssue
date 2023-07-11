# MemoryMappingIssue

Understand some unexpected behavior from huggingface datasets library.

## Setup

```
conda create -n MemoryMappingIssue python=3.10
conda activate MemoryMappingIssue
pip install torch transformers datasets tokenizers psutil pandas
```

You can get more details about the package versions from envioronment.yml.

## Usage

```
python main.py --help
python main.py --mode=BASIC [...args...]
python main.py --mode=SHARD [...args...]
python main.py --mode=BASIC --num_files=16 [...args...]
python main.py --mode=SHARD --num_files=16 [...args...]
```

## Observations

These observations are likely a result of my misuse/misunderstanding of the software at hand.

### BASIC

When running `BASIC`, the memory usage gradually increases as the dataset is processed. Expected behavior is that the memory usage should stay at some constant value as examples are read from disk, processed then released from memory. This does not cause a crash with the small dataset provided, but causes OOM issues when I try and process my entire dataset. This behavior can be observed even in the most conservative scenario, i.e., 

```
python main.py --mode=BASIC --batch_size=1 --writer_batch_size=1
```

in which case, on my 128G machine the memory usage caps at about ~18% (24G).

### SHARD

When running `SHARD`, we essentially get the same behavior. Interestingly though, if you run the program in shard mode, then cancel execution after, say, 4 iterations, then rerun the exact same command again, you will find that the dataset shards that are loaded from cache files are memory mapped properly and cause no additional memory footprint, e.g.,

#### First run

```
python main.py --mode=SHARD --batch_size=1 --writer_batch_size=1
```

Outputs

```
DATASET SIZE: 1.51G
RSS AFTER LOADING: 0.49G
RSS AFTER SHARDING: 0.49G
RSS AFTER SHARDING ITER 0: 2.02G                                                                         
RSS AFTER SHARDING ITER 1: 3.19G                                                                         
RSS AFTER SHARDING ITER 2: 4.55G                                                                         
RSS AFTER SHARDING ITER 3: 5.87G
```

#### Second run

```
python main.py --mode=SHARD --batch_size=1 --writer_batch_size=1
```

Outputs

```
DATASET SIZE: 1.51G
RSS AFTER LOADING: 0.15G
RSS AFTER SHARDING: 0.15G
Loading cached processed dataset at {PATH_TO_CACHE}.arrow
RSS AFTER SHARDING ITER 0: 0.15G
Loading cached processed dataset at {PATH_TO_CACHE}.arrow
RSS AFTER SHARDING ITER 1: 0.15G
Loading cached processed dataset at {PATH_TO_CACHE}.arrow
RSS AFTER SHARDING ITER 2: 0.16G
Loading cached processed dataset at {PATH_TO_CACHE}.arrow
RSS AFTER SHARDING ITER 3: 0.16G
RSS AFTER SHARDING ITER 4: 1.48G                                                                         
RSS AFTER SHARDING ITER 5: 3.1G
```

Notice that the first four iterations seemed to memory map very nicely and did not take up any memory, but the latter two started hogging resources.


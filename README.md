# MetagenomicsCNN

A pipeline to classify metagenomics reads by their taxonomy by using convolutional neural networks (CNN).

`get_reads` - code to download RefSeq genomes and produce simulated reads

`read2array`  - code to encode reads as 2D arrays ("images")

`cnn` - code for Convolutional Neural Networks (CNN)

# Dependencies

## Python and mason
Use `conda` to create an environment from the .yml file. Mason will be installed from bioconda.
```
conda env create -f environment.yml`
conda activate metagenomics
```

# Pipeline

1. Download genomes and produce simulated reads
2. Train CNN

## Download genomes and produce simulated reads
Code is in `get_reads`

Download bacterial genomes from RefSeq as specified in `get_reads/ncbi_ids.csv`
```
python download_refseq.py
```

Simulate Illumina reads using mason simulator
```
bash simulate_reads_mason.sh
```

Simulate Nanopore reads (50x coverage)
```
python nanopore_simulator.py 500 1 50
python nanopore_simulator.py 500 2 50
python nanopore_simulator.py 500 5 50
python nanopore_simulator.py 800 1 50
python nanopore_simulator.py 1000 1 50
python nanopore_simulator.py 1000 2 50
python nanopore_simulator.py 1000 5 50
python nanopore_simulator.py 1000 10 50
python nanopore_simulator.py 1200 1 50
```

## Train CNN
Code is in `cnn`
```
python train.py /work-zfs/mschatz1/xwang145/data/long_reads/read_500_error_1

python train.py /work-zfs/mschatz1/xwang145/data/long_reads/read_500_error_10

python train.py /work-zfs/mschatz1/xwang145/data/long_reads/read_1000_error_1

python train.py /work-zfs/mschatz1/xwang145/data/long_reads/read_1000_error_10
```
### read_500_error_1
```
13/05/2020 23:22:19 [3, 50000] loss: 0.501
Saved: /home-4/xwang145@jhu.edu/workzfs-mschatz1/xwang145/data/cnn/read_500_error_1/cnn_epoch_2.i_49999.pth
```

### read_500_error_10
```
d13/05/2020 23:22:03 [3, 10000] loss: 0.804
Saved: /home-4/xwang145@jhu.edu/workzfs-mschatz1/xwang145/data/cnn/read_500_error_10/cnn_epoch_2.i_9999.pth
```

### read_1000_error_1
```
13/05/2020 22:56:50 [2, 20000] loss: 0.423
Saved: /home-4/xwang145@jhu.edu/workzfs-mschatz1/xwang145/data/cnn/read_1000_error_1/cnn_epoch_1.i_19999.pth
```

### read_1000_error_10
```
13/05/2020 23:01:11 [2, 50000] loss: 0.578
Saved: /home-4/xwang145@jhu.edu/workzfs-mschatz1/xwang145/data/cnn/read_1000_error_10/cnn_epoch_1.i_49999.pth
```

## Test CNN
Code is in `cnn`
```

```

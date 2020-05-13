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

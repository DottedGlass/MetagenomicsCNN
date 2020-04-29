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
2. Covert reads into 2D images
3. Train CNN

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

Simulate Nanopore reads
```
python nanopore_simulator.py
```

## Convert reads into 2D images
Code is in `read2array`

```
python get_images.py ../data/mason_reads ../data/images
```

Reads belonging to a species are saved as array. Each row is the flattened image transformation of the read.

e.g.

Species A has 300 reads of length 150. After running read2array the corresponding image is 141x141. This script will save the 300 images as a 300x19881 numpy array (141x141=19881)

The data array for each species is saved in `data/images`

## Train CNN

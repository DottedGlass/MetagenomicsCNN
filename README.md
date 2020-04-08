# MetagenomicsCNN

A pipeline to classify metagenomics reads by their taxonomy by using convolutional neural networks (CNN).

`get_reads` - code to download RefSeq genomes and produce simulated reads

`read2array`  - code to encode reads as 2D arrays ("images")

`cnn` - code for Convolutional Neural Networks (CNN)

# Dependencies


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

## Covert reads into 2D images
Code is in `read2array`

```
python get_images.py ../data/mason_reads ../data/images
```

## Train CNN

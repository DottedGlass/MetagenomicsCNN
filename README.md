# MetagenomicsCNN

A pipeline to classify metagenomics reads by their taxonomy by using convolutional neural networks (CNN).

`get_reads` - code to download RefSeq genomes and produce simulated reads

`cnn` - code for Convolutional Neural Networks (CNN)

# Dependencies

## Python and mason
Use `conda` to create an environment from the .yml file. Mason will be installed from bioconda.
```
conda env create -f environment.yml
conda activate metagenomics
```

# Pipeline

1. Download genomes and produce simulated reads
2. Train CNN
3. Test CNN

## Download genomes and produce simulated reads
All data (reference genomes, simulated reads, and pytorch  models) is saved to a folder called `data` that will be generated in the root of this repo.

Download bacteria and archaea genomes from RefSeq as specified in `get_reads/ncbi_ids.csv`

Code is in `get_reads`
```
python download_refseq.py
```

### Simulate Illumina reads using mason simulator
```
bash simulate_reads_mason.sh
```

### Simulate Nanopore reads (50x coverage).
Syntax is `python nanopore_simulator [read length] [error rate] [coverage]`
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
Previous step generated long reads that will be located in `data/long_reads`. The following code trains 4 CNNs on 4 sets of long reads. 20% of the data is held out for testing. Pytorch models are saved in `data/cnn`
* Read length 500 with 1% error
* Read length 500 with 10% error
* Read length 1000 with 1% error
* Read length 1000 with 10% error

`cnn/train.py` loads all the long reads into memory. Each read in a batch is converted into a time series and then an image (Gramian Angular Field) before passed as input to the CNN for training. Conversion from reads to images during training was done instead of saving all the images first because of storage limits on our computing resource (MARCC). At 50x coverage, we need to write hundreds of thousands of images for each species, which is very memory intensive.

Code is in `cnn`

Syntax is `python train.py [path to long reads]`
```
python train.py ../data/long_reads/read_500_error_1
python train.py ../data/long_reads/read_500_error_10
python train.py ../data/long_reads/read_1000_error_1
python train.py ../data/long_reads/read_1000_error_10
```

## Test CNN
After training, you can test on the held out data.

Code is in `cnn`
Syntax is `python test.py [path to long reads] [path to model to test on]`
```
python test.py ../data/long_reads/read_500_error_1 cnn_epoch_3.i_289999.pth
python test.py ../data/long_reads/read_500_error_10 cnn_epoch_3.i_219999.pth
python test.py ..data/long_reads/read_1000_error_1 cnn_epoch_1.i_219999.pth
python test.py ../data/long_reads/read_1000_error_10 cnn_epoch_1.i_49999.pth
```

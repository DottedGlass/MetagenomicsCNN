import os
import argparse
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Simulate long reads')
    parser.add_argument('read_length', type=int, help='Length of reads')
    parser.add_argument('error_rate', type=int, help='Percent error on reads')
    parser.add_argument('coverage', type=int, help='Amount of coverage to simulate')
    args = parser.parse_args()

    # simulation parameters
    readlength = args.read_length
    error_rate = args.error_rate
    coverage = args.coverage

    # list of species
    species_list = ['NC_010117', 'NZ_LN832404', 'NC_018621', 'NC_014494', 'NC_004113', 'NC_009515', 'NC_023013', 'NC_008698', 'NC_020246', 'NC_014374']

    # directories
    refseq_dir = "../data/RefSeq/"
    out_dir = "../data/long_reads/read_" + str(readlength) + "_error_" + str(error_rate) +  '/'
    if not os.path.exists(out_dir): # create output directory if it already doesn't exist
        os.makedirs(out_dir)

    num_reads_list = []

    # run simulation
    for species in species_list:
        genome_file = refseq_dir + species + ".fasta"
        sim_reads_file = out_dir + species +  ".reads." + str(readlength) + "bp.fa"

        # compute number of reads based on coverage
        genome = str(list(SeqIO.parse(genome_file,"fasta"))[0].seq)
        num_reads = int(np.ceil(len(genome)*coverage/readlength))
        num_reads_list.append(num_reads)

        nanopore_simulator(genome_file, num_reads, readlength, error_rate, sim_reads_file, circular=True)

    # save simulation paramters to text file
    with open(out_dir + "parameters.txt", "w") as f:
        f.write("Length of reads: " + str(readlength) + "\n")
        f.write("Error rate: " + str(error_rate) + "\n")
        f.write("Coverage: " + str(coverage) + "\n")
        f.write("Number of samples\n")
        f.write("-----------------\n")
        for species, num_reads in zip(species_list, num_reads_list):
            f.write(species + ": " + str(num_reads) + "\n")

def nanopore_simulator(genome_file, num_reads, readlength, error_rate, sim_reads_file, circular=False):
    """Simulates long reads

    Parameters
    ----------
    error_rate : int or float
        Percent error.

    circular : boolean
        Determines whether or not to account for cicular DNA. Default is False.
    """

    # read in reference genome
    genome = str(list(SeqIO.parse(genome_file,"fasta"))[0].seq)
    num_errored_bases = int(np.rint(readlength*error_rate/100))

    # modify for circular genome
    if circular:
        genome = genome + genome[0:readlength]

    # perfect reads positions
    rng = np.random.default_rng()
    starting_base = rng.choice(len(genome)-readlength,size=num_reads, replace=False)

    # simulate long reads with errors
    nt2int = {'A':0, 'C':1, 'G':2, 'T':3}
    int2nt = {0:'A', 1:'C', 2:'G', 3:'T'}

    sequences = []
    for i in range(num_reads):

        # get perfect read
        sim_read = genome[starting_base[i]:(starting_base[i]+readlength)]

        # add in errors to the read
        error_position = rng.choice(readlength,size=num_errored_bases, replace=False)

        for j in range(num_errored_bases):
            # replace original nt with the nt that comes after alphabetically
            cur_error_pos = error_position[j]
            orig_nt = sim_read[cur_error_pos]

            if orig_nt not in 'ACGT': # set unknown bases to A
                error_nt = 'A'
            else:
                error_nt_int = (nt2int[orig_nt]+1) % 4
                error_nt = int2nt[error_nt_int]

            sim_read = sim_read[:cur_error_pos] + error_nt + sim_read[(cur_error_pos+1):]

        # add to list of SeqRecords
        sim_seq = Seq(sim_read, generic_dna)
        id = "simulated." + str(i+1)
        sequences.append(SeqRecord(sim_seq, id=id, name="", description=""))

    # save as fasta
    SeqIO.write(sequences, sim_reads_file, "fasta")

    print("simulation complete:",sim_reads_file)

    ## TODO: create alignment file and error file


if __name__ == "__main__":
    main()

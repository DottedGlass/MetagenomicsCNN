import os
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna

def nanopore_simulator(genome_file, num_reads, readlength, error_rate, sim_reads_file, circular=False):
    """Simulates long reads

    Parameters
    ----------
    circular : boolean
        Determines whether or not to account for cicular DNA. Default is False.
    """

    # read in reference genome
    genome = str(list(SeqIO.parse(genome_file,"fasta"))[0].seq)
    num_errored_bases = int(np.rint(readlength*error_rate))

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

    species_list = ['NC_010117', 'NZ_LN832404', 'NC_018621', 'NC_014494', 'NC_004113', 'NC_009515', 'NC_023013', 'NC_008698', 'NC_020246', 'NC_014374']

    # directories
    refseq_dir = "../data/RefSeq/"
    num_reads = 100000
    readlength = 6000
    error_rate = 0.01
    out_dir = "../data/long_reads/read_" + str(readlength) + "_error_" + str(error_rate) +  '/'

    # save simulation paramters to text file
    if not os.path.exists(out_dir): # create output directory if it already doesn't exist
        os.makedirs(out_dir)

    with open(out_dir + "parameters.txt", "w") as f:
        f.write("Number of reads: " + str(num_reads) + "\n")
        f.write("Length of reads: " + str(readlength) + "\n")
        f.write("Error rate:: " + str(error_rate) + "\n")


    # run simulation
    for species in species_list:
        genome_file = refseq_dir + species + ".fasta"
        sim_reads_file = out_dir + species +  ".reads." + str(readlength) + "bp.fa"
        # align_file = out_dir + species + ".align." + str(readlength) + ".txt"
        # error_file = out_dir + species + ".error." + str(readlength) + ".txt"
        nanopore_simulator(genome_file, num_reads, readlength, error_rate, sim_reads_file, circular=True)

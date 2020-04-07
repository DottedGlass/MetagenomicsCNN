import os
import csv

from Bio import Entrez
from Bio import SeqIO

def download(dataset_csv_path='ncbi_ids.csv', save_path='../data/RefSeq'):
    """
    Download genomes from NCBI's nucleotide database.

    Given the microbe ids given in a csv file, fetches the genomes
    corresponding to those ids from the nucleotide database.

    Args:
        dataset_csv_path: path to csv file containing the genome ids.
        save_path: save directory for the output FASTA files.

    Returns:
        None
    """

    Entrez.email = "your_email@gmail.com"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(dataset_csv_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            microbe_id = row[0].split('.')[0]
            if os.path.exists(os.path.join(save_path, microbe_id + '.fasta')):
                continue

            handle = Entrez.efetch(db="nucleotide", id=microbe_id,
                                   rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            SeqIO.write(record, os.path.join(save_path, microbe_id + ".fasta"),
                        "fasta")

if __name__ == "__main__":
    download()

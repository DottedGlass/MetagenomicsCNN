#!/usr/bin/env Rscript
# reads simulator for nanopore sequencing
nanopore_simulator = function(genomefile, reads, readlength, err, read_output,align_output, error_output){
    raw_data = readLines(genomefile)
    genome = paste(raw_data[-1], collapse = "")
    circlegenome = paste(genome,substr(genome, 1, readlength), sep='')
    length = nchar(genome)
    read_start = sample(length, reads, replace = T)
    perfect_simulate = sapply(read_start, function(x){substr(circlegenome, x, x+readlength-1)})
    perfect_reads = paste(perfect_simulate, collapse = "")
    error_position = sample(nchar(perfect_reads),floor(nchar(perfect_reads)*err))
    error_base = sample(3, floor(nchar(perfect_reads)*err), replace = T)
    replaced_base = sapply(error_position, function(x){substr(perfect_reads, x, x)})
    replaced_number = as.numeric(gsub("A", "1", gsub("C", "2", gsub("G","3", gsub("T", "4", replaced_base)))))
    simulate = perfect_reads
    replaced_base = c()
    for(j in 1:length(error_position)){
        substr(simulate, error_position[j], error_position[j])= as.character(error_base[j] + replaced_number[j])
    }
    simulate_merge = gsub("1", "A", gsub("2", "C", gsub("3","G", gsub("4", "T", gsub("5", "A", gsub("6", "C", gsub("7", "G", simulate)))))))
    error = sapply(error_position, function(x){substr(simulate_merge, x, x)})
    split.start = seq(1, nchar(simulate_merge), by = readlength)
    simulate_result = sapply(split.start, function(x){substr(simulate_merge, x, x+readlength-1)})
    read_number = error_position%/%readlength +1
    position = error_position%%readlength
    read_number[position[position==0]]=read_number[position[position==0]] -1
    position[position==0]=readlength
    replaced_base = sapply(error_position, function(x){substr(perfect_reads, x, x)})
    error_table = cbind(read_number, position, replaced_base, error)
    colnames(error_table)=c("error_read_ID", "error_position", "correct_base", "error_base")
    alignment = cbind(read_start, read_start+readlength-1)
    colnames(alignment) = c("start", "end")
    write.table(simulate_result, file = read_output, sep="\t", col.names = F)
    write.table(alignment, file = align_output, sep="\t", col.name = T)
    write.table(error_table, file = error_output, sep="\t", col.name = T)
    print(paste("results have saved as", read_output, align_output, ", and", error_output, sep = ' '))
}

# list of species
# species_list <- c('NC_010117', 'NZ_LN832404', 'NC_018621', 'NC_014494', 'NC_004113', 'NC_009515', 'NC_023013', 'NC_008698', 'NC_020246', 'NC_014374')
species_list <- c('NC_010117', 'NZ_LN832404')
# directories
out_dir <- "../data/long_reads_simulated/"
refseq_dir <- "../data/RefSeq/"

# simulation parameters
num_reads <- 100000
readlength <- 6000
error_rate <- 0.01

#  run simulations
for (species in species_list) {
  genome_file <- paste0(refseq_dir,species,".fasta")
  sim_reads_file <- paste0(out_dir,species,".reads.",readlength,"bp.fasta")
  align_file <- paste0(out_dir,species,".align.",readlength,".txt")
  error_file <- paste0(out_dir,species,".error.",readlength,".txt")

  nanopore_simulator(genome_file, num_reads, readlength, error_rate, sim_reads_file, align_file, error_file)

}

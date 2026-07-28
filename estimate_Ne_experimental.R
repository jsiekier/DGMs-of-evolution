#library(poolSeq)
library(poolSeq, lib="~/promotion/s_estimation/lib/")
#library(poolSeq, lib="~/promotion/pop_generation/lib/")
library("optparse")
library(data.table)
library(argparse)


# Create a parser
parser <- ArgumentParser()

# Add the --filename argument
#parser$add_argument("--filename", help = "Path to the file", required = TRUE)
#parser$add_argument("--animal", help = "identifier animal str", required = TRUE)
#parser$add_argument("--poolSize", help = "identifier animal str", required = TRUE)

# Parse the command-line arguments
args <- parser$parse_args()
# Use the filename
cat("Provided filename:", args$filename, "\n")



animal<-"experimental"

step_size<-10
num_replicates<-10
train_reps<-10
num_steps<-7

num_train_steps<-3
end_generation<-60
Ncensus<-1000

gen<-rep(seq(0,end_generation, by=step_size), num_replicates)
repl_tmp<-rep(1:num_replicates, each=num_steps)

file_name <- "<path_to_chrom_wise_sync>/Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID_"
out_file_name <- "solution_tmp.txt"

chroms <- c('2L', '2R', '3L', '3R', '4', 'X')
for (chrom in chroms) {
    chrom
    input_file <- paste0(file_name, chrom, '.sync')

    mySync <- read.sync(file=input_file,
                        gen=gen,
                        repl=repl_tmp,
                        polarization ="reference")

    Nes<-list()



    for (i in 1:(train_reps)){
      p0 <- paste0("F0", ".R", as.character(i))
      pt <- paste0("F", as.character((num_train_steps - 1) * step_size), ".R", as.character(i))

      beg.freq <- mySync@alleles[[paste0("F0.R", i, ".freq")]]
      end.freq <- mySync@alleles[[paste0("F", (num_train_steps - 1) * step_size, ".R", i, ".freq")]]

      beg.cov <- mySync@alleles[[paste0("F0.R", i, ".cov")]]
      end.cov <- mySync@alleles[[paste0("F", (num_train_steps - 1) * step_size, ".R", i, ".cov")]]

      element<-estimateNe(p0=beg.freq, pt=end.freq, cov0=beg.cov, covt=end.cov,
                          t=(num_train_steps-1)*step_size, Ncensus=Ncensus, poolSize=c(1000,1000)) #,method = "P.planI" #"W.planI" #"P.planI"
      print(element)
      Nes[i]<-element
      print(Nes[i])

    }
    Nes

    Ne<-sum(unlist(Nes))/length(Nes)
    print(Ne)

    array_string <- paste(unlist(Nes), collapse = ";")
    new_line <- paste(array_string, Ne, animal,chrom,  sep = "\t")
    # Check if file exists; if not, create it
    if (!file.exists(out_file_name)) {
      file.create(out_file_name)
    }

    # Read existing content
    existing_content <- readLines(out_file_name, warn = FALSE)

    # Append new line
    updated_content <- c(existing_content, new_line)

    # Write back to file
    writeLines(updated_content, out_file_name)

    tp<-seq(0,(num_train_steps-1)*step_size, by=step_size)
    tp
}




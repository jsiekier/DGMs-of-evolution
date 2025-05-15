#library(poolSeq)
#install.packages("poolSeq-0.3.5", repos=NULL, type="source")
#install.packages("poolSeq", repos=NULL, type="source", lib="~/pop_generation/lib/")
#library(poolSeq)
library(poolSeq, lib="~/pop_generation/lib/")
library("optparse")
library(data.table)
library(argparse)


# Create a parser
parser <- ArgumentParser()

# Add the --filename argument
parser$add_argument("--filename", help = "Path to the file", required = TRUE)
parser$add_argument("--animal", help = "identifier animal str", required = TRUE)

# Parse the command-line arguments
args <- parser$parse_args()
# Use the filename
cat("Provided filename:", args$filename, "\n")


# Define file name
file_name <- "solution_tmp.txt"


step_size<-5 #10 #
num_replicates<-10
train_reps<-10
num_steps<-16 #7 #
#t<-75
num_train_steps<-7 #5 #7 #11 #7#11 5
end_generation<-75 #60 #
Ncensus<-1000
poolSize<-200

gen<-rep(seq(0,end_generation, by=step_size), num_replicates)#rep(seq(0,end_generation, by=step_size), each=num_steps) #
repl_tmp<-rep(1:num_replicates, each=num_steps)# rep(seq(1,num_replicates, by=1), num_replicates) #


mySync <- read.sync(file=args$filename, #_noise_200_200
                    gen=gen,
                    repl=repl_tmp, 
                    polarization ="reference")

Nes<-list()



for (i in 1:(train_reps)){
  idx_p0<-((i-1)*num_steps*2)+7
  idx_pt<-(i*num_steps*2)+5
  idx_cov0<-((i-1)*num_steps*2)+8
  idx_covt<-(i*num_steps*2)+6
  
  p0<-as.numeric(unlist(c(mySync@alleles[,..idx_p0])))
  pt<-as.numeric(unlist(c(mySync@alleles[,..idx_pt])))
  cov0<-as.numeric(unlist(c(mySync@alleles[,..idx_cov0])))
  covt<-as.numeric(unlist(c(mySync@alleles[,..idx_covt])))
  tmp<-mySync@alleles$F0.R1.freq
  
  element<-estimateNe(p0=p0, pt=pt,
                      cov0=cov0, covt=covt,
                      t=(num_train_steps-1)*step_size, Ncensus=Ncensus, poolSize=c(poolSize, poolSize))
  print(element)
  Nes[i]<-element
  print(Nes[i])

}
Nes

Ne<-sum(unlist(Nes))/length(Nes)
Ne

array_string <- paste(unlist(Nes), collapse = ";") 
new_line <- paste(array_string, Ne, args$animal, sep = "\t")
# Check if file exists; if not, create it
if (!file.exists(file_name)) {
  file.create(file_name)
}

# Read existing content
existing_content <- readLines(file_name, warn = FALSE)

# Append new line
updated_content <- c(existing_content, new_line)

# Write back to file
writeLines(updated_content, file_name)

tp<-seq(0,(num_train_steps-1)*step_size, by=step_size)
tp




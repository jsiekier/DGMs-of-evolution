import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Subsample haplotype file")
    parser.add_argument("--haplo_file", required=True,
                        help="Input haplotype file (plain text)")
    parser.add_argument("--output_sampled", required=True,
                        help="Output path for subsampled haplotypes")
    parser.add_argument("--individuals", type=int, default=100,
                        help="Fraction of lines to keep (0–1). Default = 1.0 (keep all)")
    parser.add_argument("--seed", type=int, default=42,help="Random seed for reproducibility")
    return parser.parse_args()

def subsample_haplotypes(haplo_file, output_file, fraction, seed):
    random.seed(seed)
    kept = 0
    total = 0
    individual_ids= random.sample(range(1000),fraction)
    with open(haplo_file, "r") as infile, open(output_file, "w") as outfile:
        for line_counter, line in enumerate(infile):
            line=line.replace('\n','').split('\t')

            new_line=line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3]+'\t'
            haplos=line[4].split(' ')
            for id_index,id in enumerate(individual_ids):
                if id_index:
                    new_line+=' '
                new_line+=haplos[id]
            new_line+='\n'

            outfile.write(new_line)

    print(f"[INFO] Wrote {kept} of {total} lines to {output_file} (fraction={fraction})")

def main():
    args = parse_args()
    subsample_haplotypes(args.haplo_file, args.output_sampled, args.individuals, args.seed)

if __name__ == "__main__":
    main()
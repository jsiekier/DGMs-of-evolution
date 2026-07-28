from pathlib import Path
import argparse


def process_sync_file(input_sync, cut_sync, output_prefix):
    """
    1. Cuts the input sync file to the first 70 SNP columns.
    2. Splits the resulting file into one file per chromosome.

    Parameters
    ----------
    input_sync : str or Path
        Path to the original input .sync file.
    cut_sync : str or Path
        Path where the cut .sync file will be written.
    output_prefix : str or Path
        Prefix for the chromosome-specific output files.
        The chromosome name and '.sync' are appended automatically.
    """

    input_sync = Path(input_sync)
    cut_sync = Path(cut_sync)
    output_prefix = Path(output_prefix)

    # ---------------------------------------------------------
    # Step 1: Cut sync file to the first 70 SNP columns
    # ---------------------------------------------------------
    print(f"Cutting sync file:\n  {input_sync}\n-> {cut_sync}")

    with input_sync.open("r") as infile, cut_sync.open("w") as outfile:
        for line in infile:
            columns = line.rstrip("\n").split("\t")

            # Keep chromosome, position, reference base
            # and the first 70 SNP columns
            outfile.write("\t".join(columns[:73]) + "\n")

    print("Cutting completed.\n")

    # ---------------------------------------------------------
    # Step 2: Split cut sync file by chromosome
    # ---------------------------------------------------------
    print(f"Splitting sync file by chromosome:\n  {cut_sync}")

    current_chromosome = None
    output_file = None

    try:
        with cut_sync.open("r") as infile:
            for line in infile:
                columns = line.rstrip("\n").split("\t")
                chromosome = columns[0]

                # Open a new output file when chromosome changes
                if chromosome != current_chromosome:
                    if output_file is not None:
                        output_file.close()

                    current_chromosome = chromosome
                    output_path = Path(
                        f"{output_prefix}{current_chromosome}.sync"
                    )

                    print(f"  Writing chromosome {current_chromosome} -> {output_path}")

                    output_file = output_path.open("w")

                output_file.write(line)

    finally:
        if output_file is not None:
            output_file.close()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut a sync file to 70 columns and split it by chromosome."
    )

    parser.add_argument(
        "input_sync",
        help="Path to the original input .sync file."
    )

    parser.add_argument(
        "cut_sync",
        help="Path to the output .sync file containing the first 70 SNP columns."
    )

    parser.add_argument(
        "output_prefix",
        help="Prefix for chromosome-specific output files."
    )

    args = parser.parse_args()

    process_sync_file(
        input_sync=args.input_sync,
        cut_sync=args.cut_sync,
        output_prefix=args.output_prefix,
    )


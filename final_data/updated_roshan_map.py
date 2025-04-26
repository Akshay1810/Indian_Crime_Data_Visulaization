import argparse
import pandas as pd
import numpy as np

def expand_yearly_data(input_csv: str, output_csv: str,
                       start_year: int = 2001, end_year: int = 2013,
                       random_seed: int = None):
    """
    Reads input_csv, duplicates each unique record for every year in [start_year, end_year],
    and fills the Loss_of_Property columns with random ints between 0 and the input threshold.

    Parameters:
        input_csv: path to the source CSV (must include Year column).
        output_csv: path where the expanded CSV will be saved.
        start_year: first year in the output range (inclusive).
        end_year: last year in the output range (inclusive).
        random_seed: optional seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load the data
    df = pd.read_csv(input_csv)

    # Extract the unique schema rows (drop Year so it will be filled in)
    unique_rows = df.drop(columns=['Year']).drop_duplicates()

    # Prepare container for generated rows
    records = []

    # Generate for each year and each schema row
    for year in range(start_year, end_year + 1):
        for _, row in unique_rows.iterrows():
            # copy the base row
            new_row = row.to_dict()
            new_row['Year'] = year

            # override thresholds with random values in [0, threshold]
            for col in ['Loss_of_Property_1_25_Crores',
                        'Loss_of_Property_25_100_Crores',
                        'Loss_of_Property_Above_100_Crores']:
                threshold = row[col]
                new_row[col] = np.random.randint(0, threshold + 1)

            records.append(new_row)

    # Create DataFrame and write out
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Expanded data saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand fraud data across multiple years with random values up to given thresholds."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to write the expanded CSV.")
    parser.add_argument("--start-year", type=int, default=2001,
                        help="First year to generate (default: 2001).")
    parser.add_argument("--end-year", type=int, default=2013,
                        help="Last year to generate (default: 2013).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    expand_yearly_data(args.input_csv, args.output_csv,
                       args.start_year, args.end_year, args.seed)
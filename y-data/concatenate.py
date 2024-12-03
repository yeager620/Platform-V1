import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    final_csv_path = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/v1.2.2-game-vectors_2021-04-01_2024-10-30.csv"

    # Define the output directory and final CSV path
    # output_directory = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/"
    # os.makedirs(output_directory, exist_ok=True)

    # List to hold all csv paths
    all_csv_paths = [
        "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/game_vectors_2021-04-01_2022-12-31.csv",
        "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/game_vectors_2023-04-01_2024-12-31.csv"]

    for path in all_csv_paths:
        df = pd.read_csv(path)
        print(f"{path} has {len(df)} rows.")

    if all_csv_paths:

        all_dataframes = [pd.read_csv(csv) for csv in all_csv_paths]

        # Concatenate all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove duplicate rows if any
        # combined_df.drop_duplicates(subset=['Game_PK'], inplace=True)

        # Ensure 'Game_Date' column exists
        if 'Game_Date' in combined_df.columns:
            # Convert 'Game_Date' to datetime for proper sorting
            combined_df['Game_Date'] = pd.to_datetime(combined_df['Game_Date'])

            # Sort by 'Game_Date' in chronological order
            combined_df.sort_values(by='Game_Date', inplace=True)

            # Optionally, reset index after sorting
            combined_df.reset_index(drop=True, inplace=True)

            # Save the final concatenated DataFrame to CSV
            combined_df.to_csv(final_csv_path, index=False)
            print(f"Final CSV saved to {final_csv_path}")
            print(f"Final CSV has {len(combined_df)} rows.")
        else:
            print("Error: 'Game_Date' column not found in the combined DataFrame.")
    else:
        print("No data was processed. Final CSV not created.")


if __name__ == "__main__":
    main()

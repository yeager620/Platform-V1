import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from MaqAnalytics.VectorConstructor.DataPipeline import DataPipeline


def generate_monthly_intervals(start_date: str, end_date: str):
    """
    Generate a list of (start, end) date tuples for each month in the range.
    """
    intervals = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")

    while current_start < final_end:
        # Calculate the end of the current month
        current_end = current_start + relativedelta(months=1) - timedelta(days=1)
        if current_end > final_end:
            current_end = final_end
        intervals.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start += relativedelta(months=1)

    return intervals


def process_interval(start_end_tuple):
    """
    Process a single interval using DataPipeline and return the resulting DataFrame.
    """
    start_date, end_date = start_end_tuple
    try:
        pipeline = DataPipeline(start_date=start_date, end_date=end_date, version=2)
        result_df = pipeline.process_games()

        if result_df is not None and not result_df.empty:
            print(f"Successfully processed interval {start_date} to {end_date}")
            return result_df
        else:
            print(f"No data returned for interval {start_date} to {end_date}.")
            return pd.DataFrame()  # Return empty DataFrame if no data
    except Exception as e:
        print(f"Error processing interval {start_date} to {end_date}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def main():
    # Define the larger date range
    overall_start_date = "2023-04-01"
    overall_end_date = "2024-12-31"

    # Define the output directory and final CSV path
    output_directory = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/"
    final_csv_path = os.path.join(output_directory, f"v1.2.6-game_vectors_{overall_start_date}_{overall_end_date}.csv")
    os.makedirs(output_directory, exist_ok=True)

    # Generate month-long intervals
    intervals = generate_monthly_intervals(overall_start_date, overall_end_date)
    print(f"Total intervals to process: {len(intervals)}")

    # Define the number of parallel workers (adjust based on your system)
    max_workers = min(8, os.cpu_count() or 1)

    # List to hold all DataFrames
    all_dataframes = []

    # Process intervals in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_interval = {
            executor.submit(process_interval, interval): interval for interval in intervals
        }

        for future in as_completed(future_to_interval):
            interval = future_to_interval[future]
            try:
                df = future.result()
                if not df.empty:
                    all_dataframes.append(df)
            except Exception as exc:
                print(f"Interval {interval} generated an exception: {exc}")

    if all_dataframes:
        # Concatenate all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove duplicate rows if any
        combined_df.drop_duplicates(subset=['Game_PK'], inplace=True)

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
        else:
            print("Error: 'Game_Date' column not found in the combined DataFrame.")
    else:
        print("No data was processed. Final CSV not created.")


if __name__ == "__main__":
    main()

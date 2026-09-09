import pandas as pd
import os


def split_by_load(
    input_parquet="cwru_raw_data.parquet",
    output_directory="load_parquets"
):

    # Read the raw Parquet
    df = pd.read_parquet(input_parquet)

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Find all loads
    loads = sorted(df["motor_load_hp"].dropna().unique())

    print(f"Loads found: {loads}")

    # Create one Parquet for each load
    for load in loads:

        df_load = df[df["motor_load_hp"] == load].copy()

        output_path = os.path.join(
            output_directory,
            f"cwru_load_{load}.parquet"
        )

        df_load.to_parquet(
            output_path,
            engine="pyarrow",
            index=False
        )

        print(
            f"Load {load}: "
            f"{len(df_load)} recordings → {output_path}"
        )


if __name__ == "__main__":
    split_by_load()
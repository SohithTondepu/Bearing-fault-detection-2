import os
import scipy.io as sio
import pandas as pd
import numpy as np


def create_raw_parquet(
    metadata_path="cwru_12k_dataset.csv.xlsx",
    output_path="cwru_raw_data.parquet"
):

    df_meta = pd.read_excel(metadata_path)

    print(f"Found {len(df_meta)} recordings.")

    all_rows = []

    for _, row in df_meta.iterrows():

        mat_path = row["local_file_path"]

        if not os.path.exists(mat_path):
            print(f"Skipping missing file: {mat_path}")
            continue

        try:
            mat = sio.loadmat(mat_path)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue

        # Find vibration signals
        de_key = next(
            (k for k in mat.keys() if "DE_time" in k),
            None
        )

        fe_key = next(
            (k for k in mat.keys() if "FE_time" in k),
            None
        )

        de_data = (
            mat[de_key].flatten().astype(np.float64)
            if de_key
            else None
        )

        fe_data = (
            mat[fe_key].flatten().astype(np.float64)
            if fe_key
            else None
        )

        # Start with ALL metadata columns
        row_data = {
            "mat_file_name": row["mat_file_name"],
            "label": row["label"],
            "dataset_category": row["dataset_category"],
            "fault_location": row["fault_location"],
            "fault_component": row["fault_component"],
            "fault_diameter_inch": row["fault_diameter_inch"],
            "outer_race_position": row["outer_race_position"],
            "motor_load_hp": row["motor_load_hp"],
            "motor_speed_rpm": row["motor_speed_rpm"],
        }

        # Add raw vibration data
        row_data["de_data"] = (
            de_data.tolist() if de_data is not None else None
        )

        row_data["fe_data"] = (
            fe_data.tolist() if fe_data is not None else None
        )

        row_data["de_num_samples"] = (
            len(de_data) if de_data is not None else 0
        )

        row_data["fe_num_samples"] = (
            len(fe_data) if fe_data is not None else 0
        )

        all_rows.append(row_data)

        print(
            f"{row['mat_file_name']} | "
            f"{row['label']} | "
            f"Fault diameter: {row['fault_diameter_inch']} | "
            f"DE samples: {len(de_data) if de_data is not None else 0} | "
            f"FE samples: {len(fe_data) if fe_data is not None else 0}"
        )

    # Create dataframe
    df_raw = pd.DataFrame(all_rows)

    # Save as Parquet
    df_raw.to_parquet(
        output_path,
        engine="pyarrow",
        index=False
    )

    print(f"\nSaved {len(df_raw)} recordings to:")
    print(output_path)

    return df_raw


if __name__ == "__main__":
    create_raw_parquet()
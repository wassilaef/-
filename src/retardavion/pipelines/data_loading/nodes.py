import pandas as pd
import glob
from google.cloud import storage
import os


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Charge tous les CSV depuis un dossier GCS (écrit par Spark), les télécharge
    dans /tmp, puis concatène en un seul DataFrame.
    """
    client = storage.Client(project=project)
    bucket_name, *folder_parts = bucket_path.split("/", 1)
    prefix = folder_parts[0].rstrip("/") + "/part-"

    # Téléchargement de chaque blob CSV
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        name = blob.name.rsplit("/", 1)[-1]
        if name.endswith(".csv"):
            blob.download_to_filename(f"/tmp/{name}")

    # Lecture et concaténation
    files = glob.glob("/tmp/part-*.csv")
    if not files:
        raise ValueError("No objects to concatenate")
    dfs = [pd.read_csv(f, sep=",", index_col=None) for f in files]
    return pd.concat(dfs, ignore_index=True)


def partition_primary_csv(input_csv: str, output_folder: str, chunk_size: int) -> None:
    """
    Lit primary.csv et écrit des fichiers part-00000.csv, part-00001.csv, …
    de `chunk_size` lignes chacun dans `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(input_csv)
    num_rows = len(df)
    for start in range(0, num_rows, chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        part_id = start // chunk_size
        filename = os.path.join(output_folder, f"part-{part_id:05d}.csv")
        chunk.to_csv(filename, index=False)
        print(f"→ {filename} ({len(chunk)} lignes)")

import os
import requests
from pathlib import Path

def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    dest_dir = Path("data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "ai4i2020.csv"

    print(f"Descargando dataset desde: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_file, "wb") as f:
            f.write(response.content)
        print(f"Dataset guardado exitosamente en {dest_file} ({dest_file.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"Error descargando desde URL. Intentando usar ucimlrepo fallback...")
        try:
            from ucimlrepo import fetch_ucirepo
            ai4i2020 = fetch_ucirepo(id=601)
            df = ai4i2020.data.original
            df.to_csv(dest_file, index=False)
            print(f"Dataset guardado mediante ucimlrepo fallback en {dest_file}")
        except Exception as fallback_err:
            print(f"Fallo también el fallback: {fallback_err}")
            raise e

if __name__ == "__main__":
    download_dataset()

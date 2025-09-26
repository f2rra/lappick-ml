import os
import argparse
import kaggle


def download_kaggle_dataset(dataset_name, download_path):
    """
    Mengunduh dataset dari Kaggle ke direktori lokal.
    """

    # Pastikan direktori tujuan ada
    os.makedirs(download_path, exist_ok=True)

    try:
        print(f"Mengunduh dataset '{dataset_name}'...")
        # Perintah download dataset dari Kaggle API
        kaggle.api.dataset_download_files(
            dataset=dataset_name,
            path=download_path,
            unzip=True
        )
        print("Pengunduhan selesai.")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh dataset: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script untuk mengunduh dataset dari Kaggle."
    )

    # Tambahkan argumen untuk nama dataset
    parser.add_argument(
        'dataset_name',
        type=str,
        help="Nama dataset di Kaggle (misal: 'datasets/titanic')"
    )

    # Tambahkan argumen opsional untuk direktori tujuan
    parser.add_argument(
        '--dest',
        type=str,
        default='data/raw',
        help="Jalur folder tujuan untuk menyimpan dataset. Default: 'data/raw'"
    )

    args = parser.parse_args()

    # Jalur unduhan akan menjadi direktori saat ini ditambah folder tujuan
    destination_folder = os.path.join(os.getcwd(), args.dest)

    # Panggil fungsi unduhan dengan argumen dari terminal
    download_kaggle_dataset(args.dataset_name, destination_folder)

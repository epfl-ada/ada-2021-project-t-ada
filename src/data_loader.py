"""
Data loader functions.
"""
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

from .paths import TEST_DATA_PATH, TEST_FILENAME


def download_test_data() -> None:
    """
    Downloads the test dataset of quotes from the NYT from Google Drive.
    """
    if os.path.exists(TEST_DATA_PATH):
        print('Filename', TEST_FILENAME, 'already exists')
        return

    # Download from google drive
    gdd.download_file_from_google_drive(
        file_id='1MtCmY5zeLhdKOw8aGCgE_e5yVaODkZYW',
        dest_path=TEST_DATA_PATH,
        unzip=False,
        showsize=True,
    )

    print('Filename', TEST_FILENAME, 'downloaded')


if __name__ == '__main__':
    download_test_data()

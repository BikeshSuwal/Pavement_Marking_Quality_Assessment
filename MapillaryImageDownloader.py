import requests
import os
import time
import csv
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ACCESS_TOKEN = 'MLY|Your_mapillary_access_token'

bounding_box = (
    85.39379005883325,
    27.67127292410541,
    85.42134653713765,
    27.68086161049864
)

NUM_IMAGES = 500
IMAGE_DIR = 'mapillary_images'
os.makedirs(IMAGE_DIR, exist_ok=True)
METADATA_FILE = 'image_metadata.csv'
LOG_FILE = 'download_errors.log'

DATE_FROM = datetime(2025, 4, 2)
DATE_TO = datetime(2025, 4, 3)

session = requests.Session()

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_image_ids(bbox, total_limit, access_token):
    image_ids = []
    url = 'https://graph.mapillary.com/images'
    params = {
        'access_token': access_token,
        'fields': 'id,captured_at',
        'bbox': ','.join(map(str, bbox)),
        'limit': 500
    }
    next_url = url
    log("Fetching image IDs...")
    while next_url and len(image_ids) < total_limit:
        response = session.get(next_url, params=params)
        response.raise_for_status()
        data = response.json()

        for image in data.get('data', []):
            try:
                captured_ms = image['captured_at']
                captured_time = datetime.utcfromtimestamp(captured_ms / 1000)
                if DATE_FROM <= captured_time < DATE_TO:
                    image_ids.append(image['id'])
            except Exception as e:
                logging.error(f"Skipping image (ID parsing): {e}")

        paging = data.get('paging', {})
        next_url = paging.get('next')
        params = None
        time.sleep(0.2)

    log(f"Found {len(image_ids)} image IDs in date range.")
    return image_ids[:total_limit]

def get_image_metadata(image_id):
    url = f'https://graph.mapillary.com/{image_id}'
    params = {
        'access_token': ACCESS_TOKEN,
        'fields': 'thumb_2048_url,computed_geometry'
    }
    response = session.get(url, params=params)
    response.raise_for_status()
    return response.json()

def download_image(url, filename):
    response = session.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    return True

def process_image(image_id):
    try:
        metadata = get_image_metadata(image_id)
        image_url = metadata['thumb_2048_url']
        coordinates = metadata['computed_geometry']['coordinates']
        longitude, latitude = coordinates
        filename = os.path.join(IMAGE_DIR, f'{image_id}.jpg')
        if os.path.exists(filename):
            return 'skipped', None
        download_image(image_url, filename)
        return 'downloaded', {
            'image_id': image_id,
            'latitude': latitude,
            'longitude': longitude,
            'file_path': filename
        }
    except Exception as e:
        logging.error(f"Failed to process {image_id}: {e}")
        return 'failed', None

def main():
    # Track already downloaded image IDs
    existing_ids = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row['image_id'])

    image_ids = get_image_ids(bounding_box, NUM_IMAGES, ACCESS_TOKEN)
    image_ids = [i for i in image_ids if i not in existing_ids]

    total = len(image_ids)
    log(f"Starting download of {total} new images...")

    file_exists = os.path.exists(METADATA_FILE)
    with open(METADATA_FILE, 'a', newline='') as csv_file:
        fieldnames = ['image_id', 'latitude', 'longitude', 'file_path']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        downloaded, skipped, failed = 0, 0, 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_image, img_id): img_id for img_id in image_ids}
            for future in tqdm(as_completed(futures), total=total, desc="Processing"):
                status, result = future.result()
                if status == 'downloaded':
                    writer.writerow(result)
                    downloaded += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1

    log("Download session complete.")
    log(f"✅ Downloaded: {downloaded}")
    log(f"⏭️ Skipped: {skipped}")
    log(f"❌ Failed: {failed} (see '{LOG_FILE}')")

if __name__ == '__main__':
    main()

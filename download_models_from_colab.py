# ==============================================================================
# COPY & PASTE THIS SINGLE CELL INTO YOUR GOOGLE COLAB NOTEBOOK TO EXPORT MODELS
# ==============================================================================
import os
import shutil
import hashlib
from pathlib import Path
from google.colab import drive, files

# 1. Mount Google Drive if not already mounted
if not os.path.exists('/content/drive/MyDrive'):
    print("[Colab] Mounting Google Drive...")
    drive.mount('/content/drive')
else:
    print("[Colab] Google Drive is already mounted.")

# 2. Target directory in Google Drive
source_dir = "/content/drive/MyDrive/dataset_RE/dataset/trained_models"

if not os.path.exists(source_dir):
    print(f"[Colab] Default path '{source_dir}' not found directly.")
    print("[Colab] Searching Google Drive for trained model files...")
    for root, dirs, f_list in os.walk('/content/drive/MyDrive'):
        if any(target in f_list for target in [
            'selected_relationship_classifier.pkl',
            'final_relationship_classifier.pkl',
            'relationship_classifier.pkl'
        ]):
            source_dir = root
            print(f"[Colab] Auto-located models in: {source_dir}")
            break

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

if os.path.exists(source_dir):
    print("=" * 65)
    print(f"[Colab] Found trained models directory: {source_dir}")
    print("=" * 65)
    
    files_to_pack = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    print(f"Total model files found: {len(files_to_pack)}\n")
    for f in sorted(files_to_pack):
        full_p = os.path.join(source_dir, f)
        size_kb = os.path.getsize(full_p) / 1024
        sha = get_file_sha256(full_p)[:16]
        print(f"  * {f:<42} | {size_kb:>8.2f} KB | SHA256: {sha}...")

    # 3. Create ZIP archive
    print("\n[Colab] Creating zip archive '/content/trained_models.zip' ...")
    shutil.make_archive("/content/trained_models", "zip", source_dir)
    zip_size_mb = os.path.getsize("/content/trained_models.zip") / (1024 * 1024)
    print(f"[Colab] Archive created successfully ({zip_size_mb:.2f} MB).")

    # 4. Copy to Google Drive root as backup in case browser pop-up blocks direct download
    backup_path = "/content/drive/MyDrive/trained_models.zip"
    try:
        shutil.copy2("/content/trained_models.zip", backup_path)
        print(f"[Colab] Backup copy saved to Google Drive root: {backup_path}")
    except Exception as e:
        print(f"[Colab] Note: Drive backup notice: {e}")

    # 5. Trigger automatic browser download
    print("[Colab] Triggering automatic browser download of 'trained_models.zip' ...")
    files.download("/content/trained_models.zip")
    print("\n" + "=" * 65)
    print(">>> TRANSFER INITIATED! <<<")
    print("Your browser should now download 'trained_models.zip'.")
    print("Once downloaded, place 'trained_models.zip' into your Windows project:")
    print("    C:\\Users\\devad\\Downloads\\image-provenance-using-graph-main\\models\\")
    print("(Or leave it in your Windows Downloads folder and run verify_and_import_models.py)")
    print("=" * 65)
else:
    print(f"[Colab ERROR] Could not locate model directory under /content/drive/MyDrive.")
    if os.path.exists('/content/drive/MyDrive'):
        print("[Colab] Available top-level folders in MyDrive:")
        for item in os.listdir('/content/drive/MyDrive'):
            print(f"  - {item}")



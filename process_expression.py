import os
import pandas as pd

base_dir = r"E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\cleaned\\exp detection"
processed_dir = r"E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\processed"
os.makedirs(processed_dir, exist_ok=True)

def make_image_list(split):
    split_dir = os.path.join(base_dir, split)
    image_paths = []
    labels = []
    for emotion in os.listdir(split_dir):
        emotion_dir = os.path.join(split_dir, emotion)
        if os.path.isdir(emotion_dir):
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Store relative path to image
                    rel_path = os.path.join(split, emotion, fname)
                    image_paths.append(rel_path)
                    labels.append(emotion)
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    df.to_csv(os.path.join(processed_dir, f'expression_{split}_images.csv'), index=False)
    print(f"Saved {split} image list.")

for split in ['train', 'test']:
    make_image_list(split)

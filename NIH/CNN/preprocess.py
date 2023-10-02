import os
import shutil
import csv
from typing import List


def train_test_split_images(root_dir: str, training_fraction: float) -> None:

    # create test and training sub-directories
    try:
        os.makedirs(os.path.join(root_dir, "test_nih","images"))
        os.makedirs(os.path.join(root_dir, "training_nih","images"))
        print('Created test and training directories')
    except FileExistsError:
        print("Directory already exists!")

    # determine which images to ingest into each sub-directory
    files = os.listdir(os.path.join(root_dir, 'images'))
    split_index = int(len(files) * training_fraction)

    # Split the array into two parts at the calculated index
    train_set = files[:split_index]
    test_set = files[split_index:]

    print(f'train_set: {len(train_set)} elements')
    print(f'test_set: {len(test_set)} elements')

    source_images = os.path.join(root_dir, 'images')
    # copy training set
    for file in train_set:
        shutil.copy2(os.path.join(source_images, file), os.path.join(root_dir, 'training_nih', 'images'))

    # copy test set
    for file in test_set:
        shutil.copy2(os.path.join(source_images, file), os.path.join(root_dir, 'test_nih', 'images'))

    return train_set, test_set


def train_test_split_labels(root_dir: str, train_set: List[int], test_set: List[int]) -> None:

    # Path to the CSV file
    file_path = os.path.join(root_dir, 'sample_labels.csv')
    training_rows = []
    test_rows = []
    
    # Open the file
    with open(file_path, 'r') as file:
        # Create a reader object
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in reader:
            if row[0] in train_set:
                training_rows.append(row)
            if row[0] in test_set:
                test_rows.append(row)

    # write to training labels file
    training_labels = os.path.join(root_dir, "training_nih", "labels.csv")
    test_labels = os.path.join(root_dir, "test_nih", "labels.csv")
    with open(training_labels, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(training_rows)
    with open(test_labels, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_rows)


def filter_emphysema_rows(input_file: str, output_file: str) -> None:
    with open(input_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Read the header row

        with open(output_file, 'w', newline='') as output_csv:
            writer = csv.writer(output_csv)
            writer.writerow(['image_id', 'emphysema', *header[2:]])  # Write the header row to the output file

            for row in reader:
                image_id = row[0]
                finding_labels = row[1]

                if 'Emphysema' in finding_labels:
                    writer.writerow([image_id, 'true', *row[2:]])
                else:
                    writer.writerow([image_id, 'false', *row[2:]])


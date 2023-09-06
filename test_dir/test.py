import os
import shutil
import csv


# create test and training sub-directories
try:
    os.makedirs(os.path.join("test_nih","images"))
    os.makedirs(os.path.join("training_nih","images"))
    print('Created test and training directories')
except FileExistsError:
    print("Directory already exists!")

# determine which images to ingest into each sub-directory
files = os.listdir(os.path.join('images'))
split_index = int(len(files) * 0.8)

# Split the array into two parts at the calculated index
train_set = files[:split_index]
test_set = files[split_index:]

print(f'train_set: {train_set}')
print(f'test_set: {test_set}')

source_images = os.path.join('images')
# copy training set
for file in train_set:
    shutil.copy2(os.path.join(source_images, file), os.path.join('training_nih', 'images'))

# copy test set
for file in test_set:
    shutil.copy2(os.path.join(source_images, file), os.path.join('test_nih', 'images'))


# Path to the CSV file
file_path = os.path.join('labels.csv')
training_rows = []
test_rows = []
# Open the file
with open(file_path, 'r') as file:
    # Create a reader object
    reader = csv.reader(file)
    # Iterate over each row in the CSV file
    for row in reader:
        if f'{row[0]}.png' in train_set:
            training_rows.append(row)
        if f'{row[0]}.png' in test_set:
            test_rows.append(row)
print(f'training_rows: {training_rows}')
print(f'test_rows: {test_rows}')

# write to training labels file
training_labels = os.path.join("training_nih", "labels.csv")
test_labels = os.path.join("test_nih", "labels.csv")
with open(training_labels, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(training_rows)
with open(test_labels, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test_rows)

import os
import shutil
import argparse
import  re
os.chdir(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument(
    '--origin_dir', type=str, required=True, help="dataset path")

parser.add_argument(
    '--target_dir', type=str, required=True, help="target_dataset path")

def transform_file_deform(input_file, output_dir):
#     class_list = [
#  'cup',
#  'stairs',
#  'stool',
#  'door',
#  'person',
#  'bowl',
#  'radio',
#  'wardrobe',
#  'lamp',
#  'xbox'
# ]

    # Extracting the base name of the file to create a corresponding output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0] + '_deform'+ '.txt'
    output_path = os.path.join(output_dir, base_name)
    if 'train' in base_name:
        with open(input_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                # base = line.strip()[:-4]  # Removing the '.off' extension and any trailing newline
                base = line[:-1]
                shape_name = re.findall(r'^(.*?)_\d', base)[0]
                outfile.write(f'{base}\n')
                # if shape_name in class_list:
                outfile.write(f'{base}_ffd1\n')
                outfile.write(f'{base}_ffd2\n')
                outfile.write(f'{base}_mixup\n')
    else:
        with open(input_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                base  = line[:-1]  # Removing the '.off' extension and any trailing newline
                outfile.write(f'{base}\n')

def transform_file(input_file, output_dir):
    # Extracting the base name of the file to create a corresponding output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0] + '.txt'
    output_path = os.path.join(output_dir, base_name)
    with open(input_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                base = line[:-1] # Removing the '.off' extension and any trailing newline
                outfile.write(f'{base}\n')





def transform_all_files(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # Process each .txt file in the directory
    for file in os.listdir(input_dir):
        condition = 'train' in file or 'test' in file
        if file.endswith('.txt') and condition:
            print(file)
            transform_file_deform(os.path.join(input_dir, file), output_dir)
            transform_file(os.path.join(input_dir, file), output_dir)
        elif 'filelist' in file or 'shape' in file:
            print(file)
            target_file_path = os.path.join(output_dir, os.path.basename(file))
            source_file_path = os.path.join(input_dir, os.path.basename(file))
            shutil.copy(source_file_path, target_file_path)


if __name__ == '__main__':
    opt = parser.parse_args()

    input_dir = opt.origin_dir
    output_dir = opt.target_dir

    # Replace 'input_directory' with the path to your directory containing the .txt files
    # Specify the new directory where the transformed files will be saved
    transform_all_files(input_dir, output_dir)
    print('Done')

import os
import codecs
import re

def process_input_file(file_path):
    subjects = {}
    current_subject = None

    with codecs.open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # each line is with the following format:
            # Genesis 1:5	And God called the
            # Extract the subject and the rest as text
            m = re.match(r'^(.+?)\s+(\d+:\d+.*?)$', line)
            if m and m.group(1) and m.group(2):
                if m.group(1) not in subjects:
                    subjects[m.group(1)] = []
                subjects[m.group(1)].append(line)

    return subjects

def output_subject_files(subjects, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_number = 1
    for subject, lines in subjects.items():
        output_file = os.path.join(output_dir, f'en_{file_number}.txt')
        with codecs.open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))

        print(f"Output file created: {output_file}")
        file_number += 1

input_file_path = 'bible_english.txt'  # Replace with your input file path
output_directory = 'output'  # Replace with your desired output directory

processed_subjects = process_input_file(input_file_path)
output_subject_files(processed_subjects, output_directory)

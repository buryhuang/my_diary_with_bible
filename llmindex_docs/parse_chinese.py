import os
import codecs
import re

def process_input_file(file_path):
    subjects = {}

    with codecs.open(file_path, 'r', encoding='utf-16') as file:
        current_subject = None
        for line in file:
            line = line.strip()

            # each line is with the following format:
            # 2:15 主上帝把那人安置在伊甸園，叫他耕種，看守園子。
            if re.match(r'^(\d+:\d+.*?)', line):
                if current_subject not in subjects:
                    subjects[current_subject] = []
                subjects[current_subject].append(f"{current_subject} " + line)
            else:
                current_subject = line

    return subjects

def output_subject_files(subjects, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_number = 1
    for subject, lines in subjects.items():
        output_file = os.path.join(output_dir, f'ch_{file_number}.txt')
        with codecs.open(output_file, 'w', encoding='utf-16') as file:
            file.write('\n'.join(lines))

        print(f"Output file created: {output_file}")
        file_number += 1

input_file_path = 'bible_chinese.txt'  # Replace with your input file path
output_directory = 'output'  # Replace with your desired output directory

processed_subjects = process_input_file(input_file_path)
output_subject_files(processed_subjects, output_directory)

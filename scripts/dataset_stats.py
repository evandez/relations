from pathlib import Path
import json
import statistics

# Initial Dictionary to store the data
data_summary = {
    'total_files': 0,
    'total_samples': 0,
    'relations': {},
    'samples_per_category': {},
    'samples_per_relation': {}
}

# Lists to store samples counts for later statistics calculation
samples_per_relation_list = []

# Function to update the summary dictionary based on each file's data
def update_summary(data):
    data_summary['total_samples'] += len(data['samples'])
    
    relation = data['properties']['relation_type']
    if relation not in data_summary['relations']:
        data_summary['relations'][relation] = 0
    data_summary['relations'][relation] += 1

    if relation not in data_summary['samples_per_relation']:
        data_summary['samples_per_relation'][relation] = 0
    data_summary['samples_per_relation'][relation] += len(data['samples'])
    samples_per_relation_list.append(len(data['samples']))
    
    category = data['properties']['domain_name']
    if category not in data_summary['samples_per_category']:
        data_summary['samples_per_category'][category] = 0
    data_summary['samples_per_category'][category] += len(data['samples'])

# Recursive function to process all .json files in a directory (and subdirectories)
def process_directory(directory: Path):
    for file in directory.iterdir():
        if file.is_file() and file.suffix == '.json':
            data_summary['total_files'] += 1
            with file.open() as json_file:
                data = json.load(json_file)
                update_summary(data)
        elif file.is_dir():
            process_directory(file)

# Run the script on a specified directory
process_directory(Path('./data'))

# Calculate statistics after processing all files
data_summary['samples_stats'] = {
    'average_samples_per_relation': statistics.mean(samples_per_relation_list),
    'min_samples_per_relation': min(samples_per_relation_list),
    'max_samples_per_relation': max(samples_per_relation_list),
    'std_dev_samples_per_relation': statistics.stdev(samples_per_relation_list),
}

# Print the summarized data
print(json.dumps(data_summary, indent=4))

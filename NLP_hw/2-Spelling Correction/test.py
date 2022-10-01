import requests

FILE_URL = 'https://norvig.com/ngrams/count_1w.txt'
SAVE_PATH = 'count_1w.txt'


my_dict = {}
def save_data(file_url=FILE_URL, out_path=SAVE_PATH):
    response = requests.get(file_url)
    if (response.status_code):
        data = response.text
        with open(out_path, 'w+') as f:
            f.write(data)
    print('Data Saved at: {}'.format(out_path))

save_data()

def parse_data(src_path=SAVE_PATH):
    with open(src_path, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split('\t')[0]
            value = line.split('\t')[1].replace('\n', '')
            my_dict[key] = value
    
    print('Data Parsed! # {} keys in total'.format(len(my_dict.keys())))

if __name__ == "__main__":
    save_data()
    parse_data()
import re

def remove_links(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    cleaned_text = re.sub(url_pattern, '', text)
    return cleaned_text

def clean_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        
    with open(path.replace('.txt','_cleaned.txt'), 'w') as new_f:
        for line in lines:
            line = remove_links(line)
            if line != '\n':
                new_f.write(line)
        

if __name__ == '__main__':
    clean_dataset('reddit_short_stories.txt')
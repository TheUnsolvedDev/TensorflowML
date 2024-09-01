import os
import tqdm

MAX_LENGTH = 128

def main():
    files = os.listdir('AmItheAsshole')

    for i in range(len(files)):
        files[i] = 'AmItheAsshole/' + files[i]
    
    counter = 19889
    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            lines_together = f.readlines()
            lines_together = list(map(lambda x:x.replace('\n',''), lines_together))
            lines_together = [i for i in lines_together if len(i) > 2]
            lines_together = ' '.join(lines_together)
            lines_together = lines_together.split('.')
            lines_together = [i for i in lines_together if len(i) > 2]
            
            buffer = ''
            for line in lines_together:
                if len(buffer.split(' ')) + len(line.split(' ')) > MAX_LENGTH:
                    buffer = line
                else:
                    buffer += ' ' + line
                    buffer = buffer.replace('\n','')
                with open('new/' + str(counter)+'.txt', 'w') as f:
                    f.write(buffer)
                    counter += 1
            
    
if __name__ == "__main__":
    main()
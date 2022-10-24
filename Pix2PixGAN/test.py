import os

path = 'data/temp/'
count = 2000
for file in os.listdir('data/temp/'):
    path_name = os.path.join(path, file)
    os.rename(path_name, f'{count}.jpg')
    count += 1
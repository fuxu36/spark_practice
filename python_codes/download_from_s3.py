import boto3
import glob


s3 = boto3.client('s3')
bucket = 'farrago-datasets'
folder = 'farrago-ml/user_data/rustonalex/2019-08-29T12-23-26/trainingData.csv/'
direc = '/Users/xue/Desktop/Farrago/Datasets/Netlogx Training Data/'
key = '119c7111e05fd078d2b4017a4d48dc70723c9fef1a012a320dd3ff1885d00062'

s3_res = boto3.resource('s3')
my_bucket = s3_res.Bucket(bucket)
print('start')

for files in my_bucket.objects.filter(Prefix=folder):
    # print(files.key)
    fname = files.key.split('/')[-1]
    s3_res.meta.client.download_file(bucket, files.key, direc+fname)

extension = 'csv'
all_filenames = [i for i in glob.glob('{}*.{}'.format(direc, extension))]

with open('{}combined_csv.csv'.format(direc), 'w') as w:
    for filename in all_filenames:
        text = open(filename).read()
        w.write(text + '\n')

print('end')






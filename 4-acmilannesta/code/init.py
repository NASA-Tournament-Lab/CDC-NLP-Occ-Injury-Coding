import boto3
import os

os.environ['AWS_SHARED_CREDENTIALS_FILE'] = './AWS.txt'
s3 = boto3.Session(profile_name='default').client('s3')
for i in range(1, 16):
	s3.download_file(Bucket='acmilannesta', Key='model-oof-'+str(i+1)+'.h5', Filename='/wdata/model-oof-'+str(i+1)+'.h5')

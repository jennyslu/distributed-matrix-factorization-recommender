import boto
import json

def read_s3_data(filepath):
	'''
	Input: filepath
	Output: Pandas DataFrame
	'''

	df_chunks = pd.read_csv(filepath, chunksize=301)
	# The size of the df is really just 301. The chunksize argument/option
	# is just for practice.
	return df_chunks.get_chunk()

def s3_upload_files(bucketname, *args):
	'''
	Input: String, List of Strings
	Output: None

	With the first string as the name of a bucket on s3, upload each individual
	file from the filepaths listed in the list of strings.
	'''

	access_key, secret_access_key = get_aws_access()
	conn = boto.connect_s3(access_key, secret_access_key)

	if conn.lookup(bucket_name) is None:
	    bucket = conn.create_bucket(bucket_name, policy='public-read')
	else:
	    bucket = conn.get_bucket(bucket_name)

	for filename in args:
		key = bucket.new_key(filename)
		key.set_contents_from_filename(filename)

def get_aws_access():
	'''
	Input: None
	Output: String, String

	Read in the .json file where the aws access key and secret access key are stored.
	Output the access and secret_access_key.
	'''

	with open('/Users/sallamander/apis/access/aws.json') as f:
		data = json.load(f)
		access_key = data['access-key']
		secret_access_key = data['secret-access-key']

	return access_key, secret_access_key

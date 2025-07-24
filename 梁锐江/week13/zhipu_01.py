import os
from dotenv import load_dotenv,find_dotenv

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    print(os.getenv('api_key'))
    print(os.getenv('base_url'))

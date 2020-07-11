import sys
import requests
import os

requests.packages.urllib3.disable_warnings()

class Downloader(object):

    def __init__(self, prog=True):
        self.prog = prog
        self.config = {
            'embedding':'https://cloud.tsinghua.edu.cn/f/258e62ed039e46978717/?dl=1',
            'roberta':'https://cloud.tsinghua.edu.cn/f/f2facb2e4bbc4406bbb5/?dl=1',
            'bilstm_crf':'',
            'bert_lstm':'',
            'roberta_lstm':'',
            'REdataset':'https://cloud.tsinghua.edu.cn/f/2f2bc23d9d4d41c0b287/?dl=1',
            'dataset':'https://cloud.tsinghua.edu.cn/f/3b04bd595f494001b807/?dl=1'}

    def download(self, url, file_path):
        r = requests.get(self.config[url], stream=True, verify=False)
        total_size = int(r.headers['Content-Length'])
        temp_size = 0
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    temp_size += len(chunk)
                    f.write(chunk)
                    f.flush()
                    if self.prog:
                        done = int(50 * temp_size / total_size)
                        sys.stdout.write("\rdownloading %s : [%s%s] %d%%" % (url, '█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                        sys.stdout.flush()

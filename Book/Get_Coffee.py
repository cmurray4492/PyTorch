import urllib.request

url = 'https://pytorch.tips/coffee.jpg'
fpath = 'coffee.jpg'
urllib.request.urlretrieve(url, fpath)

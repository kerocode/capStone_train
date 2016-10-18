import urllib2  # the lib that handles the url stuff
from sys import argv

target_url = argv

#target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data";
j = open("local.txt",'w')
data = urllib2.urlopen(argv[1]) # it's a file like object and works just like a file
for line in data: # files are iterable
    j.write(line)

'''
python copy_to_local.py target_url
'''
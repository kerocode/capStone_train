import urllib2  # the lib that handles the url stuff
from sys import argv

script, filename = argv
target_url = raw_input("> ")
#target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data";
j = open("local.txt",'w')

data = urllib2.urlopen(target_url) # it's a file like object and works just like a file
for line in data: # files are iterable
    j.write(line)

'''
python copy_to_local.py target_url
'''
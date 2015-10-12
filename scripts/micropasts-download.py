#!/usr/bin/env python3
"""
Downloads micropasts data from AWS.

Usage:
    micropasts-download.py <baseurl> <towhere>
    
Options:
    <baseurl>    The URL pointing to the AWS bucket.
    <towhere>    Directory to download it to.
"""
# coding: utf-8
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import os
import shutil

# note: docopt is not part of the Python standard library. Install via "pip install docopt".
import docopt

def main():
    opts = docopt.docopt(__doc__)
    
    # In[41]:

    BUCKET_URL = opts['<baseurl>']
    TO_WHERE = opts['<towhere>']

    # At the URL above is an XML document which points to all the rest of the
    # data. We firstly need to download the data. Python comes with a web
    # grabber built in. In Python 3 this is part of the urllib.request library.
    # In Python 2 it's part of the urllib2 library.  We can use ``urlopen`` to
    # fetch a document from the Intertrons.  We now need to parse this as XML.
    # Python comes with its own XML parser. The parser takes a file-like object.
    # Fortunately the ``urlopen`` function returns one. We can thus download and
    # parse all in one go!

    with urlopen(BUCKET_URL) as f:
        document = ET.parse(f)
    document_root = document.getroot()

    # The document root is a ``<ListBucketResult>`` element which has each file
    # listed in a separate ``<Contents>`` element. Each element is namespaced
    # which would be a pain to type out each time. Hence we'll create a custom
    # prefix ``s3`` for Amazon S3 elements.

    ns = { 's3': 'http://s3.amazonaws.com/doc/2006-03-01/' }

    # We can now use the namespace prefix to match elements without having to
    # specify the full URL which is tiresome. Test this by counting the number
    # of ``Contents`` tags:

    contents = document_root.findall('s3:Contents', ns)
    assert len(contents) > 0 # Throws an exception if there are no files to download!

    # Each ``Contents`` element has a ``Key`` element which stores the actual
    # path to the file. Use a list comprehension to extract a list of keys:
    keys = [c.find('s3:Key', ns).text for c in contents]
    print('Number of keys: {}'.format(len(keys)))

    # Now we need to form a URL for each key. Again Python has us covered.
    # ``urllib.parse`` has a handy ``urljoin`` function which does the hard work
    # for us. (In Python 2 this is in the ``urlparse`` module I *think*.)

    urls = [urljoin(BUCKET_URL, key) for key in keys]


    # OK, getting there. We now need to download all of the files. If the URL
    # ends in a ``/``, it's a directory. If it doesn't we'll assume it's a file.
    for key in keys:
        # Get the URL for this key
        url = urljoin(BUCKET_URL, key)

        # Compute the filename to download this to
        path = os.path.join(TO_WHERE, key)
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)

        # Skip URLs corresponding only to directories
        if url.endswith('/'):
            continue

        # Does the destination directory exist? If not, make it so (#1).
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # Now we need to download the file...
        print('Downloading {}...'.format(url))
        with urlopen(url) as src, open(path, 'wb') as dst:
            shutil.copyfileobj(src, dst)


if __name__ == '__main__':
    main()

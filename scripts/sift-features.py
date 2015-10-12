#!/usr/bin/env python3
"""
Find and compute SIFT features and descriptors from a directory of images.

Usage:
    sift-features.py (-h | --help)
    sift-features.py [options] <imgdir> <maskdir> <outdir>

Options:
    -h, --help              Show usage summary.
    -q, --quiet             Log only warnings and errors.

    --lowe                  Output data in format compatible with the original
                            SIFT utility by David Lowe.

    --imglob=PATTERN        Pattern to match images [default: *.JPG]
    --maskformat=FORMAT     Format to construct mask filenames relative to mask
                            directory given image filenames.
                            [default: {basename}_mask{ext}]
    --outformat=FORMAT      Format to construct feature filenames relative to
                            output directory given image filenames.
                            [default: {basename}_features.{format}]

Notes:
    The SIFT features are computed using the SIFT implementation in OpenCV's
    xfeatures2d framework with default parameters. Output is in the form of a
    compressed NumPy .npz file per image. This file contains the absolute path
    to the original image, the absolute path to the original and the keypoints
    and descriptors.

    In addition, sha256 hashes of the raw image and mask files are saved to aid
    in tracking which files gave which output.

    Some sample Python code which could be used to load the results is as
    follows. Replace 'some_filename.npz' as appropriate.

        import cv2
        import numpy as np

        contents = np.load('some_filename.npz')

        def make_kp(kp_location, kp_data, kp_attrs):
            x, y = kp_location
            data = dict(('_' + a, v) for a, v in zip(kp_attrs, kp_data))
            return cv2.KeyPoint(x, y, **data)

        keypoints = [
            make_kp(loc, dat, contents['keypoint_fields']) for loc, dat in
            zip(contents['keypoint_locations'], contents['keypoint_data'])
        ]
        descriptors = contents['descriptors']

    The keypoints and descriptors variables now hold output compatible with that
    returned from the detectAndCompute() method on OpenCV feature detectors.

    If the --lowe option is specified then data is instead saved in the ASCII
    format used by David Lowe's original SIFT demo program at [1]. This option
    is intended to allow this script to act as a free-er replacement to the
    original sift program.

    [1] http://www.cs.ubc.ca/~lowe/keypoints/

"""
import glob
import gzip
import hashlib
import logging
import os

import docopt
import numpy as np
from PIL import Image

# This requires the xfeatures2d module from OpenCV-contrib.
# See: https://github.com/itseez/opencv_contrib
from cv2.xfeatures2d import SIFT_create

def sha256_file(pn):
    """Compute the SHA256 hash of the file with pathname *pn* and return the
    digest as a hex-string.
    """
    h = hashlib.sha256()
    with open(pn, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def save_npz(out_pn, kps, descs, metadata):
    """Save keypoints and descriptors in NPZ format.
    """
    kp_attrs = ('angle', 'class_id', 'octave', 'response', 'size')
    output = {
        'descriptors': descs,
        'keypoint_fields': kp_attrs,
        'keypoint_data': np.asarray([
            [getattr(k, a) for a in kp_attrs]
            for k in kps
        ], dtype=np.float32),
        'keypoint_locations': np.asarray([
            k.pt for k in kps
        ], dtype=np.float32),
    }

    output.update(metadata)

    # Save output
    logging.info('Saving NPZ format output to %s...', out_pn)
    np.savez_compressed(out_pn, **output)

def save_lowe(out_pn, kps, descs, metadata):
    """Save output in David Lowe's sift demo program format. Note that Lowe's
    format does not support the saving of metadata :(.

    """
    # comments taken from Lowe's description.
    with open(out_pn, 'w') as f:
        # The file format starts with 2 integers giving the total number of
        # keypoints and the length of the descriptor vector for each keypoint
        # (128).
        f.write('{} {}\n'.format(len(kps), descs.shape[1]))
        for kp, desc in zip(kps, descs):
            # Then the location of each keypoint in the image is specified by 4
            # floating point numbers giving subpixel row and column location,
            # scale, and orientation (in radians from -PI to PI).
            f.write('{} {} {} {}\n'.format(
                kp.pt[0], kp.pt[1], kp.size, np.deg2rad(kp.angle - 180)
            ))

            # Finally, the invariant descriptor vector for the keypoint is given
            # as a list of 128 integers in range [0,255].
            f.write(' '.join(str(v) for v in desc))
            f.write('\n')

def main():
    # Parse command line and configure logging level
    opts = docopt.docopt(__doc__)
    logging.basicConfig(level=logging.WARN if opts['--quiet'] else logging.INFO)

    feat_detector = SIFT_create()

    # Determine which save function to use.
    save_fn, out_format = save_npz, 'npz'
    if opts['--lowe']:
        save_fn, out_format = save_lowe, 'key'

    im_glob = os.path.join(opts['<imgdir>'], opts['--imglob'])
    for im_pn in glob.glob(im_glob):
        logging.info('Processing %s...', im_pn)

        # Construct the mask and features path
        im_bn, im_ext = os.path.splitext(os.path.basename(im_pn))
        mask_pn = os.path.join(opts['<maskdir>'], opts['--maskformat'].format(
            basename=im_bn, ext=im_ext
        ))
        out_pn = os.path.join(opts['<outdir>'], opts['--outformat'].format(
            basename=im_bn, ext=im_ext, format=out_format
        ))

        # Compute hashes of image and mask
        im_hash = sha256_file(im_pn)
        mask_hash = sha256_file(mask_pn)

        # Load the image and mask from disk
        im = np.asarray(Image.open(im_pn).convert('L'))
        mask = np.asarray(Image.open(mask_pn).convert('L'))

        # Detect keypoints and features
        kps, descs = feat_detector.detectAndCompute(im, mask)

        metadata = {
            'image_path': os.path.abspath(im_pn),
            'mask_path': os.path.abspath(mask_pn),
            'image_sha256': im_hash,
            'mask_sha256': mask_hash,
        }

        # Save output
        save_fn(out_pn, kps, descs, metadata)

if __name__ == '__main__':
    main()

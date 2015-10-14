#!/usr/bin/env python3
"""
Utility to mask image.

Usage:
    mask-image.py (-h | --help)
    mask-image.py [--background=COLOR] [--quiet] <image> <mask> <output>

Options:
    -h, --help                  Show a brief usage summary.
    -q, --quiet                 Log only errors and warnings.

    -b, --background=COLOUR     Replace background with COLOR. [default: 128]

Notes:
    This utility replaces all background pixels with mid-grey. It can be used to
    pass the masked imagery to structure from motion (SfM) tools which do not
    natively support masks. This is essentially a hack to work around
    less featureful tools.

    Any EXIF metadata in the input image are copied into the output image. This
    is important as such metadata often includes important properties like focal
    length and camera model which may be used to estimate camera intrinsics.

    The background colour can be specified either as a single integer which will
    be used for all colour channels or as a comma-separated list of three
    integers which will be used for the corresponding colour channels.

    You may wish to use these colours as a guide:

        "Green" screen  41,244,24
        "Blue" screen   24,71,241

"""
import logging

import docopt
import numpy as np
from PIL import Image

def parse_colour(cv):
    """Parse cv as a single integer to be broadcast over a triple or
    comma-separated integer triple. Return an integer triple in all cases. If
    *cv* cannot be parsed as any of these formats, raise ValueError.

    """
    try:
        return 3 * [int(cv)]
    except ValueError:
        l = [int(v) for v in cv.split(',')]
        if len(l) != 3:
            raise ValueError('Colour must be specified as a triple')
        return l

def main():
    # Parse command line and configure logging level
    opts = docopt.docopt(__doc__)
    logging.basicConfig(level=logging.WARN if opts['--quiet'] else logging.INFO)

    bg_color = parse_colour(opts['--background'])

    # Load input images
    im = Image.open(opts['<image>']).convert('RGB')
    im_exif = im.info.get('exif', None) # get any EXIF metadata
    im = np.asarray(im)
    mask = np.asarray(Image.open(opts['<mask>']).convert('L'))

    # Mask each channel
    output = np.zeros_like(im)
    for c_idx, c in enumerate(bg_color):
        output[..., c_idx] = np.where(mask > 128, im[..., c_idx], c)

    # Write image to output
    Image.fromarray(output).save(opts['<output>'], exif=im_exif)

if __name__ == '__main__':
    main()

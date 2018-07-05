#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts


def main(opt):
    translator = make_translator(opt, report_score=True)
    _, attns_info, oov_info, copy_info = translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)
    
    # currently attns_info,oov_info only contain first index data of batch
    return attns_info, oov_info, copy_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)

    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)

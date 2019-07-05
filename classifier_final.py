#!/usr/bin/env python
import numpy as np
import os
import sys
import shutil
import tempfile
from argparse import ArgumentParser
import pickle

import features


def split_scanpaths(data):
    starts = np.where(data['idx'] == 0)[0]
    ends = np.append(starts[1:], len(data))
    assert starts.shape == ends.shape
    return [data[start:end][['x', 'y', 'duration']] for start, end in zip(starts, ends)]


def get_SAM_saliency_prediction_path(args, SAM_path='saliency_attentive_model/', verbose=False):
    expected_out = '{sam}/predictions/{fname}'.format(sam=SAM_path, fname=os.path.basename(args.image_file))
    if args.do_not_recomupte_saliency and os.path.exists(expected_out):
        return expected_out
    tmp_dir = tempfile.mkdtemp()
    shutil.copy(args.image_file, tmp_dir)

    cmd = 'cd {sam} ; '\
          'THEANO_FLAGS=device=cuda,floatX=float32 ; '\
          'python main.py test "{inp}/" '.format(sam=SAM_path, inp=tmp_dir)
    if verbose:
        print >> sys.stderr, cmd
    os.system(cmd)
    shutil.rmtree(tmp_dir)  # clean up
    return expected_out


def main(args):
    scanpaths = np.genfromtxt(args.scanpath_file, names=True, delimiter=',', dtype=np.float, case_sensitive='lower')
    scanpaths = split_scanpaths(scanpaths)

    mdl = pickle.load(open(args.pretrained_model))

    SAM_predictions = get_SAM_saliency_prediction_path(args)
    feat, feat_names = features.extract_features(scanpaths, args.image_file, SAM_saliency_path=SAM_predictions,
                                                 return_names=True)

    feature_ids = [feat_names.index(name) for name in mdl['feature_set']]
    feat = np.array(feat)[:, feature_ids]
    if args.mode == 'class':
        res = mdl['model'].predict(feat)
        cls_map = {0: '0', 1: '1'}
        res = map(lambda x: cls_map[x], res)
    else:
        res = mdl['model'].predict_proba(feat)[:, 1]  # probability for the positive class
        res = map(str, res)

    out_file = sys.stdout
    if args.output is not None:
        out_file = open(args.output, 'w' if not args.append else 'a')
    for val in res:
        print >> out_file, val
    out_file.close()


def parse_args():
    parser = ArgumentParser('ASD/TD classifier',
                            description='This ')
    parser.add_argument('--mode', choices=['class', 'prob'], default='prob',
                        help='Output either class labels ("TD" or "ASD") -- in the "class" mode or '
                             'the probability for the ASD class -- in the "prob" mode (default).')
    parser.add_argument('--pretrained-model', '--model', default='pretrained_model.pkl',
                        help='The pickle\'d dictionary file with the "model" field containing '
                             'the pre-trained model (withg .predict_proba() and .predict() function). '
                             'The "feature_set" field describes the features used for classification.')
    parser.add_argument('--do-not-recomupte-saliency', '--efficient', '--eff', action='store_true',
                        help='If passed, this argument will force not re-computing the SAM '
                             'saliency predictions when corresponding files already exist in SAM `predictions` folder')
    parser.add_argument('--output', '--out', '-o',
                        help='A text file where to write the output. If non passed, will print to stdout. '
                             'By default, will overwrite the file. If this is not desired, pass --append as well.')
    parser.add_argument('--append', '-a', action='store_true',
                        help='Append to the output file rather than overwriting it.')
    parser.add_argument('scanpath_file', help='A .txt file with one or more scanpaths')
    parser.add_argument('image_file', help='The stimulus image corresponding to the scanpath(s)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
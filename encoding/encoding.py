import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging

from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.ridge import bootstrap_ridge
from config import  REPO_DIR, EM_DATA_DIR



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# witch subject to use
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature", type=str, required=True)
	#how many sessions there are
	parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
	parser.add_argument("--trim", type=int, default=5)
	parser.add_argument("--ndelays", type=int, default=4)
	# nboots is probably the epochs
	parser.add_argument("--nboots", type=int, default=50)
	parser.add_argument("--chunklen", type=int, default=40)
	parser.add_argument("--nchunks", type=int, default=125)
	parser.add_argument("--singcutoff", type=float, default=1e-10)
	parser.add_argument("-use_corr", action="store_true")
	parser.add_argument("-single_alpha", action="store_true")
	logging.basicConfig(level=logging.INFO)


	args = parser.parse_args()
	globals().update(args.__dict__)

	fs = " ".join(_FEATURE_CONFIG.keys())
	assert args.feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert np.amax(args.sessions) <= 5 and np.amin(args.sessions) >=1, "1 <= session <= 5"

	sessions = list(map(str, args.sessions))
	with open("em_data/sess_to_story.json", "r") as f:

		sess_to_story = json.load(f) 
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)
	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
	allstories = list(set(train_stories) | set(test_stories))

	save_location = os.path.join("results", args.feature, args.subject)
	print("Saving encoding model & results too:", save_location)
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	downsampled_feat = get_feature_space(args.feature, allstories)
	print("Stimulus & Response parameters:")
	print("trim: %d, ndelays: %d" % (args.trim, args.ndelays))

	# Delayed stimulus
	delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, args.trim, args.ndelays)
	print("delRstim: ", delRstim.shape)
	delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat, args.trim, args.ndelays)
	print("delPstim: ", delPstim.shape)

	# Response
	zRresp = get_response(train_stories, args.subject)
	print("zRresp: ", zRresp.shape)
	zPresp = get_response(test_stories, args.subject)
	print("zPresp: ", zPresp.shape)

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print("Ridge parameters:")
	print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
		args.nboots, args.chunklen, args.nchunks, args.single_alpha, args.use_corr))

	wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
		delRstim, zRresp, delPstim, zPresp, alphas, args.nboots, args.chunklen,
		args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha,
		use_corr=args.use_corr)

	# Save regression results.
	np.savez("%s/weights" % save_location, wt)
	np.savez("%s/corrs" % save_location, corrs)
	np.savez("%s/valphas" % save_location, valphas)
	np.savez("%s/bscorrs" % save_location, bscorrs)
	np.savez("%s/valinds" % save_location, np.array(valinds))
	print("Total r2: %d" % sum(corrs * np.abs(corrs)))

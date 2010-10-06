#!/usr/bin/env python
# audio to audio alignment using particle filtering

import numpy as np
from scikits.audiolab import Sndfile, Format
from optparse import OptionParser

def KL(x1,x2) :
	pass
	return 0

def obsprob(x1,x2) :
	return KL(x1,x2)

def readwavefile(inputwav):
	f = Sndfile(inputwav, 'r')
	fs = f.samplerate
	if fs != 44100 :
		print 'only 44.1kHz filess are supported at present'
		exit(1)
	nc = f.channels
	if nc != 1 :
		print 'only 1 channel supported at present'
		exit(1)
	nframes =  f.nframes
	wav = f.read_frames(nframes, dtype=np.float32)
	f.close()
	return wav

if __name__ == '__main__' :
	parser = OptionParser(usage = 'usage: %prog [options] audio1 audio2')
	parser.add_option("--ns", dest="ns", help='number of particles', default=200)
	parser.add_option("--fft", dest="fftlen", help='analysis window length', default=2048)
	(options, args) = parser.parse_args()
	if len(args) != 2 :
		parser.print_help()
		exit(1)
	ns = options.ns
	fftlen = options.fftlen
	DT = fftlen / 44100.
	# load audio files
	audio1 = readwavefile(args[0])
	audio1 = audio1[:len(audio1) - (len(audio1) % fftlen)]
	audio2 = readwavefile(args[1])
	audio2 = audio2[:len(audio2) - (len(audio2) % fftlen)]
	# reshape audio files so that each row is an audio frame
	audio1 = np.reshape(audio1, (-1, fftlen))
	audio2 = np.reshape(audio2, (-1, fftlen))
	# init pf
	Rp = 5.
	Rt = 0.3
	sigma2p = 2.
	sigma2t = 1.
	w = np.ones(ns) / ns	# weights
	po = np.random.normal(0,1,ns)	# old position
	to = np.random.normal(1,np.sqrt(0.5),ns) # old tempo (playback speed ratio)
	# main loop
	totframes = len(audio1)
	for k in range(totframes) :
		# sample position of new particles
		pn = po + to*DT + np.random.uniform(-Rp/2, Rp/2, ns)
		tn = to + np.random.uniform(-Rt/2, Rt/2, ns)
		




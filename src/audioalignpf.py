#!/usr/bin/env python
# audio to audio alignment using particle filtering
import numpy as np
from scikits.audiolab import Sndfile
from optparse import OptionParser

def KL(x1,x2) :
	pass
	return 0

def obsprob(x1,x2) :
	#return KL(x1,x2)
	return np.dot(x1 / np.sum(x1),x2 / np.sum(x2))

def transprob(xo, xn):
	global DT
	global sigma2p
	global sigma2t
	po,to = xo
	pn,tn = xn
	return np.exp(-0.5*(((pn - po - to*DT)**2)/(sigma2p*DT) + ((tn-to)**2)/(sigma2t*DT)))

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

def resample(p,t,w) :
	ns = len(p)
	c = np.zeros(ns+1)
	for i in range(1,ns+1) :
		c[i] = c[i-1] + w[i-1]
	pp = np.zeros(ns)
	tt = np.zeros(ns)
	for i in range(ns) :
		j = 0
		r = np.random.uniform()
		while j < ns+1 and r > c[j] :
			j += 1
		j -= 1
		pp[i] = p[j]
		tt[i] = t[j]
	ww = np.ones(ns) / ns
	return (pp,tt,ww)

if __name__ == '__main__' :
	# test resample
	'''
	p = np.array([1,2,3])
	t = np.array([1,2,3])
	w = np.array([0.5,0.25,0.25])
	c = [0,0,0]
	for i in range(100) :
		a,b,x = resample(p,t,w)
		for j in range(len(a)) :
			if a[j] == 1 :
				c[0] += 1
			if a[j] == 2 :
				c[1] += 1
			if a[j] == 3 :
				c[2] += 1
	print c
	exit(1)
	'''
	# actual script
	parser = OptionParser(usage = 'usage: %prog [options] audio1 audio2')
	parser.add_option("--ns", dest="ns", help='number of particles', default=200)
	parser.add_option("--fft", dest="fftlen", help='analysis window length', default=2048)
	(options, args) = parser.parse_args()
	if len(args) != 2 :
		parser.print_help()
		exit(1)
	ns = int(options.ns)
	fftlen = int(options.fftlen)
	DT = fftlen / 44100.
	# load audio files
	audio1 = readwavefile(args[0])
	audio1 = audio1[:len(audio1) - (len(audio1) % fftlen)]
	audio2 = readwavefile(args[1])
	audio2 = audio2[:len(audio2) - (len(audio2) % fftlen)]
	# reshape audio files so that each row is an audio frame then do FFT
	audio1 = np.reshape(audio1, (-1, fftlen))
	audio2 = np.reshape(audio2, (-1, fftlen))
	audio1 = np.abs(np.fft.fft(audio1))**2
	audio2 = np.abs(np.fft.fft(audio2))**2
	# init pf
	Rp = 2.
	Rt = 0.1
	sigma2p = 2.
	sigma2t = .5
	w = np.ones(ns) / ns	# weights
	po = np.random.normal(0,1,ns)	# old position
	to = np.random.normal(1,np.sqrt(0.1),ns) # old tempo (playback speed ratio)
	# main loop
	totframes = len(audio1)
	for k in range(totframes) :
		# sample position of new particles
		pn = po + to*DT + np.random.uniform(-Rp/2, Rp/2, ns)
		tn = to + np.random.uniform(-Rt/2, Rt/2, ns)
		# compute obs. probabilities for all needed frames
		frames2observe = np.floor(pn / DT)
		for i in range(len(frames2observe)) :
			if frames2observe[i] < 0 :
				frames2observe[i] = 0.
			if frames2observe[i] > len(audio2)-1  :
				frames2observe[i] = len(audio2)-1
		f2obsprob = {}
		for f in set(frames2observe) :
			f2obsprob[f] = obsprob(audio1[k], audio2[f])
		# compute transition probability and multiply weights, along with obs. prob.
		for i in range(ns) :
			OP = f2obsprob[frames2observe[i]]
			TP = transprob((po[i], to[i]), (pn[i], tn[i]))
			w[i] *= OP * TP
		w = w/np.sum(w)
		# need resampling?
		neff = 1./sum(w**2)
		#print (k,neff)
		if neff < 2 :
			pn,tn,w = resample(pn,tn,w)
		# printout alignment
		print '%10.4f,%10.4f' % (k*DT, np.sum(pn*w))
		# switch pointers
		po,pn = pn,po
		to,tn = tn,to
		
		



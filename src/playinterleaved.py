#!/usr/bin/env python
# Create a new wav file, based on the alignment, that
# alternates excerpts between the two input files
import numpy as np
from optparse import OptionParser
from audioalignpf import readwavefile
from scikits.audiolab import Sndfile, Format
import csv

if __name__ == '__main__' :
	parser = OptionParser(usage = 'usage: %prog [options] audio1 audio2 alignment output')
	parser.add_option("-i", dest="interval", help='duration of audio excerpts', default=2.)
	(options, args) = parser.parse_args()
	if len(args) != 4 :
		parser.print_help()
		exit(1)
	# read alignment
	reader = csv.reader(open(args[2],'r'))
	alignment = []
	for line in reader :
		alignment.append((float(line[0]), float(line[1])))
	print alignment
	# read wave files
	audio1 = readwavefile(args[0])
	audio2 = readwavefile(args[1])
	out = np.zeros(max(len(audio1),len(audio2)))
	currenttime = 0
	currentaudio = 1
	currentouttime = 0
	while True :
		try :
			# read audio and write
			if currentaudio == 1 :
				excerpt = audio1[currenttime*44100:(currenttime+options.interval)*44100]
			else :
				excerpt = audio2[currenttime*44100:(currenttime+options.interval)*44100]
			out[currentouttime*44100:(currentouttime+options.interval)*44100] = excerpt
			currentouttime += options.interval
			# move pointer and swap song
			if currentaudio == 1 :
				tp = 0
				while alignment[tp][0] < currenttime + options.interval :
					tp += 1
				currenttime = alignment[tp][1]
				currentaudio = 2
			else :
				tp = 0
				while alignment[tp][1] < currenttime + options.interval :
					tp += 1
				currenttime = alignment[tp][0]
				currentaudio = 1
		except :
			print 'FIXME this sucks'
			break
	format = Format('wav')
	f = Sndfile(args[3], 'w', format, 1, 44100)
	f.write_frames(out)
	f.close()

		

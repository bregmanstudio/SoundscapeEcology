"""
Analyze species-specific patterns in environmental recordings

SoundscapeEcology methods:
		load_audio()   - load sample of a soundscape recording
		sample_audio_dir() - load group sample from multiple recordings
		analyze()      - extract per-species time-frequency partitioning from loaded audio
		visualize()    - show component spectrograms 
		resynthesize() - sonify summary statistics information
		model_fit_resynhesize() - generative statistical model of time-shift kernels
		summarize()    - show soundscape ecology entropy statistics

SoundscapeEcology static methods:
		batch_analyze() - multiple analyses for a list of recordings
		entropy() - compute entropy (in nats) of an acoustic feature distribution
		gen_test_data() - generate an artificial soundscape for testing

Workflows:
	 [load_audio(), sample_audio_dir()] -> analyze() -> [visualize(), resynthesize(), summarize()]

Copyright (c) Michael A. Casey, Bregman Media Labs, Dartmouth College, USA
License: MIT License, see attached licence file
"""

import sys, os.path, subprocess, glob, pdb
from bregman import features, sound, plca, testsignal
import numpy as np
import matplotlib.pyplot as plt

class SoundscapeEcology(object):
	"""Analyze species-specific patterns in environmental recordings
	"""
	default_plt_args = {'normalize':1, 'dbscale':1, 'cmap':plt.cm.hot, 'vmax':0.0, 'vmin':-45}
	default_feature_params = {
		'feature': 'cqft',
		'hi': 16000,
		'intensify': False,
		'lcoef': 1,
		'lo': 31.25,
		'log10': False,
		'magnitude': True,
		'nbpo': 12,
		'ncoef': 10,
		'nfft': 4096,
		'nhop': 2048,
		'onsets': False,
		'power_ext': '.power',
		'sample_rate': 44100,
		'verbosity': 0,
		'wfft': 4096,
		'window': 'hann'
	}

	def __init__(self, verbose=1, extractor=features.LogFrequencySpectrum, **kwargs):
			"""
			Initialize analysis parameters
			inputs:
					verbose - how much information to give about processing [1]
				extractor - feature extractor class  [bregman.features.LogFrequencySpectrum]
				 **kwargs - key word args for feature parameters and plot functions
			"""
			self.verbose=verbose
			self.file_list=[]
			self.sr = None
			self.channels = None
			self.x = []
			self.sample_dur = None
			self.num_samples = None
			self.frames_per_sample = None
			self.sample_points = None
			self.features = None
			self.extractor = extractor
			self.is_complex = False
			self.stereo = False
			self.n_components = None
			self.feature_params = SoundscapeEcology.default_feature_params.copy() 
			pop_dict(self.feature_params, kwargs)
			dummy = self.extractor()
			self.feature_params['feature'] = dummy.feature_params['feature']
			self.plt_args = SoundscapeEcology.default_plt_args.copy()
			pop_dict(self.plt_args, kwargs)
			self._message("",1)
			return

	def _message(self,s, l, cr=True):
		if self.verbose >= l:
			 print s,
			 if cr:
					print
			 sys.stdout.flush()

	def load_audio(self, file_name, num_samples=None, frames_per_sample=8, sample_offset=10.0):
		"""
		Extract samples from an audio file.
		inputs:
			 file_name - full path to audio file
			 num_samples - how many independent samples to draw, None=all [10]
			 frames_per_sample  - number of feature frames per sample [8]
			 sample_offset - skip from start in seconds [10.0]
		"""
		tmp_flag = False
		pth, nm = os.path.split(file_name)
		nm, ext = os.path.splitext(nm)
		self.file_list.append(file_name)
		self._message("Loading file %s"%file_name, 1, False)
		if ext!='.wav':
			file_name = self._convert_to_wav(file_name)
			tmp_flag = True
		sndfile = sound.Sndfile(file_name, 'r')
		self._check_params(sndfile, num_samples, frames_per_sample, sample_offset)
		sample_dur = self.sample_dur
		duration = sndfile.nframes / float(sndfile.samplerate)
		self._message(", duration=%3.2fs"%duration,1)
		actual_num_samples = 0 if num_samples is None else num_samples
		self._message("Sampling file, num_samples=%d, frames_per_sample=%d, sample_dur=%2.3f, total_dur=%2.3f"
									%(actual_num_samples, self.frames_per_sample, sample_dur, actual_num_samples*sample_dur), 2)
		starts = np.arange(sample_offset, duration-sample_dur, sample_dur) * self.feature_params['sample_rate']
		if num_samples is not None and num_samples>0:
			starts = np.random.permutation(starts)[:num_samples]
			starts.sort() # back into original order, now with gaps
		sample_points = self.sample_points
		for s in starts:
			s = int(round(s))
			N = min(sample_points, sndfile.nframes-s)
			x, sr, fmt = sound.wavread(file_name, first=int(round(s)), last=N)
			if N < sample_dur * sndfile.samplerate:
				x = np.r_[x, np.zeros( ( sample_points - N, sndfile.channels) )]
			if len(x) != sample_points:
				print "DEBUG: len(x) != sample_points"
				pdb.set_trace()
			if len(x.shape)<2: # fix-up MONO signals (FIXME in bregman.sound)
				x = np.atleast_2d(x).T
			self.x.append(x)
		sndfile.close()
		if tmp_flag and file_name[:4]==os.path.sep+'tmp': # double check
			if subprocess.call(['rm', file_name]) != 0:
				raise IOError("Cannot remove temporary WAV file %s"%file_name)

	def _check_params(self, sndfile, num_samples, frames_per_sample, sample_offset):
		if self.sr is None:
			self.sr = sndfile.samplerate
		if self.channels is None:
			self.channels = sndfile.channels
		if self.num_samples is None:
			self.num_samples = num_samples
		if self.frames_per_sample is None:
			self.frames_per_sample = frames_per_sample
		if self.sample_points is None:
			wfft = self.feature_params['wfft']
			nhop = self.feature_params['nhop']
			self.sample_points = (self.frames_per_sample-1) * (wfft-nhop) + wfft
		if self.sample_dur is None:
			self.sample_dur = self.sample_points / float(self.sr)
		if sndfile.samplerate != self.sr:
			raise ValueError("%s sample rate does not match previous loaded audio"%self.file_list[-1])
		if sndfile.samplerate != self.feature_params['sample_rate']:
			raise ValueError("%s sample rate does not match self.feature_params"%self.file_list[-1])
		if sndfile.channels != self.channels:
			raise ValueError("%s channels does not match previous loaded audio"%self.file_list[-1])
		duration = sndfile.nframes / float(self.sr)
		if num_samples != self.num_samples:
			raise ValueError("%s num_samples does not match previous loaded audio"%self.file_list[-1])
		if self.num_samples is not None and duration < self.num_samples * self.sample_dur:
			raise ValueError("%s duration < total sample duration"%self.file_list[-1])
		if sample_offset >= duration:
			raise ValueError("%s sample_offset >= duration"%self.file_list[-1])
				 
	def _convert_to_wav(self, file_name):
		# utility function to read an mp3 using wav read
		# returns new name of file
		pth, nm = os.path.split(file_name)
		nm, ext = os.path.splitext(nm)
		tmp_name = os.path.sep + 'tmp' + os.path.sep + nm + '.wav'
		if subprocess.call(['mplayer', '-novideo', '--really-quiet', '-ao', 'pcm:waveheader:file=%s'%tmp_name, file_name]) != 0:
			raise IOError("Cannot execute mplayer convert for %s"%file_name)
		return tmp_name
						
	def sample_audio_dir(self, dir_expr, **ldaud_kwargs):
		"""
		Pool samples from files matching dir_expr
		inputs:
			dir_expr - path/pattern string for audio files
			**ldaud_kwargs - load_audio key-word args
		"""
		file_list = glob.glob(dir_expr)
		if len(file_list) == 0:
			print "WARNING: no files found"
			return
		for f in file_list:
			self.load_audio(f, **ldaud_kwargs)

	def analyze(self, num_components=16, win=None, dual_channel=False, stereo=False, **kwargs):
		"""
		Perform feature analysis and SI-PLCA-2D on loaded audio
		Reconstruct approximate spectrogram and individual component spectrograms
		inputs:
		num_components - number of 2D plca components to extract
		           win - SI-PLCA-2D kernel shape [None=(nbpo, frames_per_sample)]
		  dual_channel - use components with relative phase of complex STFT
	 	        stereo - process both channels [L,R] of a stereo recording
	 	      **kwargs - key word arguments for Features() and Feaures.separate()
		outputs:
				self.F.{w,z,h} - SIPLCA2 w,z,h components
				self.X       - original spectrogram
				self.X_hat   - original spectrogram reconstruction from all components
				self.X_hat_l - list of component spectrogram reconstructions
		"""
		if not len(self.x):
			print "Usage: load_audio() must be called before analyze()"
			return
		pop_dict(self.feature_params, kwargs)
		self.feature_params['sample_rate'] = self.sr
		self._message("Extracting %s, nfft=%d, wfft=%d, nhop=%d"%(self.feature_params['feature'], self.feature_params['nfft'],self.feature_params['wfft'], self.feature_params['nhop']),1)
		if len(self.x[0].shape)<2 or not stereo:
			self.features = self.extractor(np.vstack(self.x), **self.feature_params)
		elif len(self.x[0].shape)>1 and stereo:
			self.stereo = True
			self.features = self.extractor(np.vstack(self.x)[:,0], **self.feature_params)
			self.features_R = self.extractor(np.vstack(self.x)[:,1], **self.feature_params)
			self.F_R = self.features_R
		self.F = self.features
		if dual_channel:
			U = self.F._phase_map()
			self.F.X = np.r_[self.F.X, U] # relative phase feature matrix
			self.is_complex = True
		if win is None:
			win=(self.feature_params['nbpo'], self.frames_per_sample)
		self._message("Extracing SIPLCA2, num_components=%d, win=(%d,%d)"%(num_components,win[0], win[1]), 1)
		self.F.separate(plca.SIPLCA2, n=num_components, win=win, **kwargs)
		self._message("Inverting %d extracted components"%len(self.F.z),1)
		self.X_hat_l = self.F.invert_components(plca.SIPLCA2, self.F.w, self.F.z, self.F.h)
		self.X_hat = np.array(self.X_hat_l).sum(0)
		self.X = self.F.X # for convenience
		self.n_components = len(self.X_hat_l)
		if self.stereo:
			kwargs['alphaZ'] = 0.0 # reset alphaZ, we want the same number of components in each channel
			self._message("Extracing RIGHT SIPLCA2, num_components=%d, win=(%d,%d)"%(self.F.z.shape[0],win[0], win[1]), 1)
			self.F_R.separate(plca.SIPLCA2, n=self.n_components, win=win, initW=self.F.w, initZ=self.F.z, initH=self.F.h, **kwargs)
			self._message("Inverting RIGHT %d extracted components"%len(self.F_R.z),1)
			self.X_R_hat_l = self.F.invert_components(plca.SIPLCA2, self.F.w, self.F.z, self.F.h)
			self.X_R_hat = np.array(self.X_R_hat_l).sum(0)
			self.X_R = self.F_R.X # for convenience
	 
	def visualize(self, plotXi=True, plotX=False, plotW=False, plotH=False, **pargs):
		"""
		Plot reconstructed component spectrograms
		inputs:
				 plotXi - visualize individual reconstructed component spectra [True]
				 plotX - visualize original (pre-analysis) spectrum and reconstruction [False]
				 plotW - visualize component time-frequency kernels [False]
				 plotH - visualize component shift-time activation functions [False]
				**pargs - plotting key word arguments [**self.plt_args]
		"""
		if self.X_hat_l is None: # check data exists
			print "Warning: self.analyze() must be called first"
			return
		pop_dict(self.plt_args, pargs)
		if plotX:
			self._subplots([self.X, self.X_hat], "Original Spectrum and Component Reconstruction", **self.plt_args)
			if self.stereo:
				self._subplots([self.X_R, self.X_R_hat], "Original RIGHT Spectrum and Component Reconstruction", **self.plt_args)
		if plotXi:
			self._subplots(self.X_hat_l, "Individual Component Reconstructions", **self.plt_args)
			if self.stereo:
				self._subplots(self.X_R_hat_l, "Individual RIGHT Component Reconstructions", **self.plt_args)
		if plotW:
			self._subplots(self.F.w.swapaxes(0,1), "Frequency-Time Kernels", **self.plt_args)
			if self.stereo:
				self._subplots(self.F_R.w.swapaxes(0,1), "RIGHT Frequency-Time Kernels", **self.plt_args)
		if plotH:
			self._subplots(self.F.h, "Shift-Time Activation Functions", **self.plt_args)
			if self.stereo:
				self._subplots(self.F_R.h, "RIGHT Shift-Time Activation Functions", **self.plt_args)

	def _subplots(self, X, ttl, subttl=None, **plt_args):
		plt.figure()
		rn = int(np.ceil(np.sqrt(len(X)))) # post-analysis n_components
		f = self.F # feature object instance
		for i, x in enumerate(X):
			plt.subplot(rn,rn,i+1)
			features.feature_plot(x, nofig=1, **plt_args)
			plt.title('Component %d'%i, fontsize=12)
			if f._have_cqft and x.shape[0]==len(f._logfrqs):
				plt.yticks(np.arange(0,x.shape[0],16),f._logfrqs[::16].round().astype('i'))
				if i%rn==0:
					plt.ylabel('Freq (Hz)', fontsize=12)
			elif x.shape[0]==len(f._fftfrqs):
				plt.yticks(np.arange(0,x.shape[0],100),f._fftfrqs[::100].round().astype('i'))
				if i%rn==0:
					plt.ylabel('Freq (Hz)', fontsize=12)
			else:
				if i%rn==0:
					plt.ylabel('Shift (steps)', fontsize=12)
			if i>=len(X)-rn:
				fr = float(f.sample_rate) / f.nhop
				plt.xticks(np.arange(0,x.shape[1],int(fr)), np.arange(0,x.shape[1]/fr,1).astype('i'))
				plt.xlabel('Time (secs)', fontsize=12)
			else:
				plt.xticks([])
		if f._have_cqft:
			ttl = ttl + ' (logFreq, dB)'
		plt.suptitle(ttl, fontsize=16)
	 
	def resynthesize(self, k, rand_phase=False, **kwargs):
		"""
		resynhesize reconstructed component k to audio
		inputs:
				k - the component to resynthesize
				rand_phase - use random instead of original STFT phases [False]
				**kwargs - key word arguments for bregman.Features.inverse()
		"""
		if self.X_hat_l is None or k>=self.n_components:
			print "k exceeds number of components"
			return
		Phi = (np.random.rand(*self.F.STFT.shape)*2-1)*np.pi if rand_phase else None
		if self.is_complex:
			nc = self.F.STFT.shape[0]
			U = self.X_hat_l[k][-nc:,:]
			X = self.X_hat_l[k][:-nc,:]
			if not rand_phase:
				Phi = self.F._phase_rec(U)
		else:
			X = self.X_hat_l[k]
			if self.stereo:
				X_R = self.X_R_hat_l[k]
				Phi_R = (np.random.rand(*self.F.STFT.shape)*2-1)*np.pi if rand_phase else None
		if self.stereo:
			return (self.F.inverse(X, Phi_hat=Phi, **kwargs), self.F_R.inverse(X_R, Phi_hat=Phi_R, **kwargs)) 
		else:
			return self.F.inverse(X, Phi_hat=Phi, **kwargs)

	def model_fit_resynthesize(self, k, distH=None, num_frames=None, lamH=100, lenH=11, sigmaH=1, returnH=False):
		"""
		Experimental: fit SIPLCA2 components with statistical model and resynthesize
		inputs:
				k     - which component to fit and resynthesize
				distH - the model distribution [None=np.random.poisson]
		 num_frames - number of resynthesis frames
				lamH  - lambda parameter multiplication factor [100]
				lenH  - smoothing window length [11]
			 sigmaH - smoothing window spread [1]
		outputs:
				The reconstructed modeled component [timeFuncs if returnH is true, else spectrogram]
		"""
		distH = np.random.poisson if distH is None else distH
		h = self.F.h[k,:,:]
		num_frames = h.shape[1] if num_frames is None else num_frames
		if num_frames<lenH:
			print "Warning: num_frames<lenH"
			return
		m = h.mean(1) # Poiss lambdas from mean over kernel shifts
		h = np.vstack([distH(lam,num_frames) for lam in m*lamH]) # Poisson texture
		hh = np.array([np.convolve(b,testsignal.gauss_pdf(lenH,lenH/2,sigmaH),'same') for b in h])
		hh = hh / hh.sum()
		if self.stereo:
			h_R = self.F_R.h[k,:,:]
			m_R = h_R.mean(1) # Poiss lambdas from mean over frequency of kernel
			h_R = np.vstack([distH(lam,h.shape[1]) for lam in m_R*lamH]) # Poisson texture
			hh_R = np.array([np.convolve(b,testsignal.gauss_pdf(lenH,lenH/2,sigmaH),'same') for b in h_R])
			hh_R = hh_R / hh_R.sum()
		if returnH:
			if self.stereo:
				return hh, hh_R
			else:
				return hh
		else:
			X_hat = self.F.invert_component(plca.SIPLCA2, self.F.w[:,k,:], self.F.z[k], hh)
			x_hat = self.F.inverse(X_hat, Phi_hat=np.random.rand(self.F.STFT.shape[0],num_frames)*2*np.pi-np.pi)
			if self.stereo:
				X_R_hat = self.F_R.invert_component(plca.SIPLCA2, self.F_R.w[:,k,:], self.F_R.z[k], hh_R)
				x_r_hat = self.F_R.inverse(X_R_hat, Phi_hat=np.random.rand(self.F_R.STFT.shape[0],num_frames)*2*np.pi-np.pi)
				return x_hat, x_r_hat
			else:
				return x_hat

	def summarize(self, show=False, plotting=True, **kwargs):
		"""
		Print and return summary statistics for analyzed components
		output: dict {      
			 'Hx':Hx, '_Hx':"entropy of original time-frequency distribution",
			 'Hx_hat':Hx_hat, '_Hx_hat':"entropy of reconstructed time-frequency distribution",
			 'z':z, '_z':"component probability distribution",
			 'Hz':Hz, '_Hz':"entropy of component probability distribution",
			 'Hxi':Hxi, '_Hxi':"entropies of component individual time-frequency reconstructions",
			 'Hwi':Hwi, '_Hwi':"entropies of (frequency,time) kernels",
			 'Hhi':Hhi, '_Hhi':"entropies of activation (shift, amplitude) functions"
			 }
		"""
		if self.X_hat_l is None:
			return
		Hx = self.entropy(self.X, **kwargs)
		Hx_hat = self.entropy(self.X_hat, **kwargs)
		z = self.F.z
		Hz = self.entropy(z)
		Hxi = np.array([self.entropy(X, **kwargs) for X in self.X_hat_l])
		Hwi = np.array([self.entropy(W, **kwargs) for W in self.F.w.swapaxes(0,1)])
		Hhi = np.array([self.entropy(H, **kwargs) for H in self.F.h])
		d = {
			'Hx':Hx, '_Hx':"entropy of original time-frequency distribution",
			'Hx_hat':Hx_hat, '_Hx_hat':"entropy of reconstructed time-frequency distribution",
			'z':z, '_z':"component probability distribution",
			'Hz':Hz, '_Hz':"entropy of component probability distribution",
			'Hxi':Hxi, '_Hxi':"entropies of component individual time-frequency reconstructions",
			'Hwi':Hwi, '_Hwi':"entropies of (frequency,time) kernels",
			'Hhi':Hhi, '_Hhi':"entropies of activation (shift, amplitude) functions"
		}
		if plotting:
			plt.figure()
			for i,k in enumerate(['z','Hxi','Hwi','Hhi']):
				plt.subplot(4,1,i+1)
				self._stemplot(d[k],d['_'+k])
				plt.xticks([])
			plt.suptitle('Distribution Entropy Hz=%3.3f'%d['Hz'], fontsize=16)
		return d

	@staticmethod
	def _stemplot(z, t):
		plt.stem(range(len(z)),z)
		plt.axis('tight')
		plt.axis(xmin=-0.1, xmax=len(z)-1+.1)
		plt.title(t, fontsize=14)

	@staticmethod
	def entropy(x, clip_val=0.001):
		"""
		Compute the entropy (in nats) of the acoustic feature distribution in x
		Cutoff values at clip_val, use only non-clipped values for entropy
		inputs:
			 x - the data to be measured (will be normalized to a distribution)
			 clip_val - the decibel threshold for measurement [0.001 = -60dB]
		outputs:
			 entropy (nats) of re-normalized distribution after thresholding at clip_val
		"""      
		x = np.array(x)
		if x.sum() < 10e-6:
			return 0.0
		x = x / x.sum() # normalize
		z = x[np.where(x>clip_val)]
		if z.sum() < 10e-6:
			return 0.0
		z = z / z.sum() # re-normalize
		return -1 * (z * np.log(z)).sum()
		
	@staticmethod
	def batch_analyze(dir_expr, **kwargs):
		"""
		SoundscapeEcology multi-analysis of files matching dir_expr
		inputs:
			dir_expr - path/pattern string for audio files
			**kwargs - keyword args for __init__(), load_audio(), and analyze()
		outputs:
			 List of SoundscapeEcology objects containing analyses
		"""
		ldaud_args={'num_samples':10, 'frames_per_sample':8, 'sample_offset':10.0}
		pop_dict(ldaud_args, kwargs)
		flist=glob.glob(dir_expr)
		flist.sort()
		sse_l = []
		for f in flist:
			sse = SoundscapeEcology(**kwargs)
			sse.load_audio(f, **ldaud_args)
			sse.analyze(**kwargs)
			sse_l.append(sse)
		return sse_l

	@staticmethod
	def gen_test_data(nc=5, nb=1, win=(12,8), dur=3.0, sz=0.5, sw=0.5, sh=0.5, bw=0.5, bh=0.5,extractor=features.LogFrequencySpectrum, **kwargs):
		"""
		Make a synthetic component dataset with given properties:
		inputs:
			 nc - num latent components [5]
			 nb - num background components [1]
			win - size of SIPLCA2D (shifts, duration) frequency-time-shift kernels [(12,8)]
			dur - duraion in seconds of dataset [3.0]
			 sz - sparseness of component Z distribution (0..1) [0.5]
			 sw - sparseness of component W distribution (0..1) [0.5]
			 sh - sparseness of component H distribution (0..1) [0.5]
			 bw - exponential sparseness rolloff for W components [0.5]
			 bh - exponential sparseness rolloff for H components [0.5]
		extractor - bregman.Features class [LogFrequencySpectrum]
			**kwargs - bregman.Features keyword args [SoundscapeEcology.default_feature_params]
		outputs: Features F
			 F.w  - plca w kernels (nFreq, nc, win[0])
			 F.z  - plca z distribution (nc)
			 F.h  - plca h functions (nc, win[1], nT)
	    F.X_hat_l - latent component spectra
		  F.X_hat - audio spectrum reconstruction
		"""
		feature_params = SoundscapeEcology.default_feature_params.copy()
		pop_dict(feature_params, kwargs)
		x = testsignal.noise(num_points=44100*dur)
		F = extractor(x, **feature_params)
		F.separate(plca.SIPLCA2, nc, win=win)
		F.invert_components(plca.SIPLCA2, F.w, F.z, F.h)
		poiss = lambda k, n: np.array(np.random.poisson(k*10, size=n), dtype='float')
		chisquare = lambda k, n: np.array(np.random.chisquare(k*10, size=n), dtype='float')
		rolloff = lambda a, b, c: c * np.exp( -b * a )
		F.z = chisquare(sz, len(F.z))
		F.z /= F.z.sum()
		nFreqs, nWtimes, nShifts, nHtimes = F.w.shape[0], F.w.shape[2], F.h.shape[1], F.h.shape[2]
		for i in range(len(F.z)):
			print "[%d] sw=%3.5f, sh=%3.5f"%(i, rolloff(i, bw, sw), rolloff(i, bh, sh))
			for t in range(nWtimes):
					freqs = poiss( rolloff(i, bw, sw), nFreqs)
					F.w[:,i,t] = freqs
			for t in range(nHtimes):
					shifts = poiss( rolloff(i, bh, sh), nShifts)
					F.h[i,:,t] = shifts
			F.w[:,i,:] /= F.w[:,i,:].sum()
			F.h[i,:,t] /= F.h[i,:,:].sum()         
		s = SoundscapeEcology()
		s.F = F
		F.X_hat_l = F.invert_components(plca.SIPLCA2, F.w, F.z, F.h)
		s.X_hat_l = F.X_hat_l
		s.n_components = len(s.X_hat_l)
		F.X = np.array(F.X_hat_l).sum(0)
		F.inverse()
		s.visualize(plotXi=True,plotW=True,plotH=True)
		return s

def pop_dict(a,b):
	"""
	Update values in dict a from dict b with same keys
	Ignore keys in b that are not in a
	return:
			a - dictinoary a modified in-place
	"""
	for k in a.keys(): a[k] = b.pop(k, a[k])
	return a

def plot_all(sse):
	"""
	Convenience function to visualize components and entropies for list of soundscape analysis objects
	"""
	d = []
	for s in sse: 
		print s.file_list[0]
		sys.stdout.flush()
		s.visualize(plotXi=True, plotX=True, plotW=True, plotH=True)
		d.append(s.summarize(clip_val=0.0005))
		print "Entropy Analysis"
		for k in ['Hx','Hx_hat','Hz','Hxi','Hwi','Hhi']:
			print d[-1]['_'+k]+':', d[-1][k]
		sys.stdout.flush()
		plt.show()
	return d

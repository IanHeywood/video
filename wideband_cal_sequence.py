from vidscp import *
import mqt
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-m','--ms',dest='myms',help='Measurement Set to process')
(options,args) = parser.parse_args()
myms = options.myms

# Make nterms = 3 MFS image of full band
# EXPORT FITS?!
#~ img1 = 'img_'+myms

#~ # Make initial LSM

#~ img1fits = img1+'.image.tt0.fits'
#~ img1alpha = img1+'.image.alpha.fits'
#~ img1alphaerr = img1+'.image.alpha.error.fits'
#~ img1beta = img1+'.image.beta.fits'

#~ gaul1 = makeSkyModel(img1fits,10.0,5.0)
#~ lsm1 = tiggerConvert(gaul1)
#~ spilsm1 = lsm1.replace('.lsm.html','.spi.lsm.html')
#~ # Make [alpha,beta] LSM

#~ addAlphaBeta(img1alpha,img1alphaerr,img1beta,lsm1,spilsm1)

#~ # Phase cal per SPW on spectral LSM
spilsm1 = 'img_sb25575669_VIDEO_XMM_1.ms.image.tt0.pybdsm.spi.lsm.html'

resetcal = True

for spw in range(0,16):
		spw = str(spw)
		info('Spectral window '+spw)
		a_fmep = myms+'/G_ampl_spw'+spw+'.fmep'
		p_fmep = myms+'/G_phase_spw'+spw+'.fmep'
		if resetcal:
			redinfo('Deleting fmeps')
			os.system('rm -rf '+p_fmep) # Zap non-standard tables
			os.system('rm -rf '+a_fmep)
			os.system('rm -rf '+myms+'/G_phase.fmep') # And standard tables just in case
			os.system('rm -rf '+myms+'/G_ampl.fmep')
	#	os.system('mkdir '+p_fmep)
	#	os.system('mkdir '+a_fmep)
		mqt.run(script='calico-wsrt-tens.py',
			job='cal_G_phase',
			section='VID_phasecal',
			config='tdlconf.profiles',
			args=['ms_sel.msname='+myms,
			'tiggerlsm.filename='+spilsm1,
			'ms_sel.ddid_index='+spw,
			'cal_g_phase.g_phase.nondefault_meptable='+p_fmep,
			'cal_g_phase.g_ampl.nondefault_meptable='+a_fmep,
			'do_output=CORR_RES'])
# 	mqt.run(script='turbo-sim.py',job='simulate',section='fake_src',config='tdlconf.profiles',args=['gridded_sky.center_source_flux='+s_flux,'ms_sel.ddid_index='+str(spw),'ms_sel.msname='+targetms])

	

# Reimage


#~ img1 = 'img_'+myms+'_5k_postflag'
#~ img2 = 'img_'+myms+'_5k_stefcal_ap_8_8_spi_resid'
#~ img3 = 'img_'+myms+'_5k_stefcal_ap_8_8_spi'
#~ oplsm = 'master_'+myms+'.pybdsm.lsm.html'
#~ spw = myms.split('_')[-1].split('.')[0].replace('spw','')
#~ gaintab = myms+'/gain.cp'

#~ ref_freq = 1.032 # centre of spw0 in GHz
#~ ref_cell = 1.0 # arcsec appropriate for spw0
#~ nchan,freqs = getSpectralInfo(myms,spw)
#~ idx = int(nchan/2)
#~ cfreq = freqs[0][idx]/1e9 # GHz
#~ cell = 1.0*(ref_freq/cfreq)
#~ im_cell_ref = str(cell)+'arcsec'

#~ # make postflag image
#~ imageMSmfs(myms,img1,im_npix_ref,im_cell_ref,im_niter,im_wplanes,im_robust_ref,'',True)
#~ # Make shallow LSM
#~ lsm1 = makeSingleSpectralSkyModels(img1+'.image.tt0.fits',10.0,3.0)
#~ # stefcal on shallow LSM, produce residuals
#~ mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal_smooth_res',config='tdlconf.profiles',args=['tiggerlsm.filename='+lsm1,'ms_sel.msname='+myms,'ms_sel.ddid_index='+str(spw),'stefcal_gain.table='+gaintab])
#~ # image residuals
#~ imageMSmfs(myms,img2,im_npix_ref,im_cell_ref,im_niter,im_wplanes,im_robust_ref,'',True)
#~ # Make deep LSM
#~ lsm2 = makeSingleSpectralSkyModels(img2+'.image.tt0.fits',5.0,3.0)
#~ # Add deep LSM to shallow LSM
#~ lsm3 = mergeLSMs([lsm1,lsm2],oplsm)
#~ # stefcal on master LSM
#~ mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal_smooth',config='tdlconf.profiles',args=['tiggerlsm.filename='+lsm3,'ms_sel.msname='+myms,'ms_sel.ddid_index='+str(spw),'stefcal_gain.table='+gaintab])
#~ # rflag
#~ rflagMS(myms,5.0,5.0,'corrected')
#~ # Final image
#~ imageMSmfs(myms,img3,im_npix_ref,im_cell_ref,10000,im_wplanes,im_robust_ref,'',True)

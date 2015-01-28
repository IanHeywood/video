from vidscp import *

myms = 'sb26444701_VIDEO_VLA9_pr2_spw1.ms'
img1 = 'img_'+myms+'_5k_postflag'
img2 = 'img_'+myms+'_5k_stefcal_ap_8_8_spi_resid'
img3 = 'img_'+myms+'_5k_stefcal_ap_8_8_spi'
oplsm = 'master_'+myms+'.pybdsm.lsm.html'
spw = myms.split('_')[-1].split('.')[0].replace('spw','')
gaintab = myms+'/gain.cp'

ref_freq = 1.032 # centre of spw0 in GHz
ref_cell = 1.0 # arcsec appropriate for spw0
nchan,freqs = getSpectralInfo(myms,spw)
idx = int(nchan/2)
cfreq = freqs[0][idx]/1e9 # GHz
cell = 1.0*(ref_freq/cfreq)
im_cell_ref = str(cell)+'arcsec'

# make postflag image
imageMSmfs(myms,img1,im_npix_ref,im_cell_ref,im_niter,im_wplanes,im_robust_ref,'',True)
# Make shallow LSM
lsm1 = makeSingleSpectralSkyModels(img1+'.image.tt0.fits',10.0,3.0)
# stefcal on shallow LSM, produce residuals
mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal_smooth_res',config='tdlconf.profiles',args=['tiggerlsm.filename='+lsm1,'ms_sel.msname='+myms,'ms_sel.ddid_index='+str(spw),'stefcal_gain.table='+gaintab])
# image residuals
imageMSmfs(myms,img2,im_npix_ref,im_cell_ref,im_niter,im_wplanes,im_robust_ref,'',True)
# Make deep LSM
lsm2 = makeSingleSpectralSkyModels(img2+'.image.tt0.fits',5.0,3.0)
# Add deep LSM to shallow LSM
lsm3 = mergeLSMs([lsm1,lsm2],oplsm)
# stefcal on master LSM
mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal_smooth',config='tdlconf.profiles',args=['tiggerlsm.filename='+lsm3,'ms_sel.msname='+myms,'ms_sel.ddid_index='+str(spw),'stefcal_gain.table='+gaintab])
# Final image
imageMSmfs(myms,img2,im_npix_ref,im_cell_ref,10000,im_wplanes,im_robust_ref,'',True)

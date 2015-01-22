from vidscp import *
import os

xx = glob.glob('*spw0*.ms')

print xx

for myms in xx:
	tt = tables.table(myms)
	if tt.nrows() == 0:
		print 'No rows, skipping',myms
		tt.done()
	else:
		tt.done()
		imgname = 'img_'+myms+'_5k_stefcal_ap_8_8_spi_dual_lsm'
#		imgname = 'img_'+myms+'_5k_postflag'
		if os.path.exists(imgname+'.fits'):
			redinfo(imgname+'.fits exists, skipping')
		else:
			info(myms+' --> '+imgname)
			im_niter = 10000 # for final corr_data!!!
			imageMSmfs(myms,imgname,im_npix_ref,im_cell_ref,im_niter,im_wplanes,im_robust_ref,'',True)
	

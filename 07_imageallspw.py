from vidscp import *
import os

xx = glob.glob('*spw11.ms')

print xx

for myms in xx:
	tt = tables.table(myms)
	if tt.nrows() == 0:
		print 'No rows, skipping',myms
		tt.done()
	else:
		tt.done()
		imgname = 'img_'+myms+'_6k_stefcal_20x8'
		if os.path.isfile(imgname+'.fits'):
			print imgname+'.fits exists, skipping'
		else:
			imageMSmfs(myms,imgname,im_npix_ref,im_cell_ref,6000,im_wplanes,im_robust_ref,'',False)
	

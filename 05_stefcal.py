import Pyxis
import ms,mqt
import glob
import os

msList = glob.glob('*spw*.ms')

for myms in msList:
	spw = int(myms.split('_')[-1].rstrip('.ms').lstrip('spw'))
	mylsm = glob.glob('master*spw'+str(spw)+'*lsm.html')
	if len(mylsm) != 0:
		mylsm = mylsm[0]
		gaintab = myms+'/gain.cp'
		info('Calibrating '+myms+', spectral window '+str(spw))
		info('Against '+mylsm)
		info('With solutions going to '+gaintab)
		mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal',config='tdlconf.profiles',args=['tiggerlsm.filename='+mylsm,'ms_sel.msname='+myms,'ms_sel.ddid_index='+str(spw),'stefcal_gain.table='+gaintab])

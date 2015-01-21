import Pyxis
import ms,mqt
import glob
import os

JOBS = 1
JOB_STAGGER = 10

#~ def calMS():
	#~ v.MS = MS
	#~ spw = int(MS.split('_')[-1].rstrip('.ms').lstrip('spw'))
	#~ v.DDID = spw
	
	#~ mylsm = glob.glob('master*spw'+str(spw)+'*lsm.html')
	#~ if len(mylsm) != 0:
		#~ mylsm = mylsm[0]
		#~ gaintab = v.MS+'/gain.cp'
		#~ info(str(v.MS)+' '+str(v.DDID))
		#~ info(gaintab)
		#~ mqt.run(script='calico-stefcal.py',job='stefcal',section='VID_stefcal',config='tdlconf.profiles',args=['tiggerlsm.filename='+mylsm,'ms_sel.msname='+v.MS,'ms_sel.ddid_index='+str(v.DDID),'stefcal_gain.table='+gaintab])

#~ def run():
	#~ v.MS_List = glob.glob('*spw*.ms')
	#~ per('MS',calMS)
	
#~ run()

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
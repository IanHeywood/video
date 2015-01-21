import Pyxis
import ms,mqt
import glob
import os

def resetCal():
	xx = glob.glob(MS.rstrip('/')+'/*.fmep')
	for item in xx:
		syscall = 'rm -rf '+item
		os.system(syscall)

def mergeMS(outputms):
	syscall = 'merge-ms.py '+outputms
	for ms in v.MS_List:
		syscall+= ' '+ms
	print syscall

def calMS():
	v.MS = MS
	spw = int(MS.split('_')[-1].rstrip('.ms').lstrip('spw'))
	v.DDID = spw
	mylsm = glob.glob('master*spw'+str(spw)+'*lsm.html')
	if len(mylsm) != 0:
		mylsm = mylsm[0]
		print v.MS, v.DDID
		mqt.run(script='calico-wsrt-tens.py',job='cal_G_phase',section='VID_phasecal',config='tdlconf.profiles',args=['tiggerlsm.filename='+mylsm,'ms_sel.msname='+v.MS,'ms_sel.ddid_index='+str(spw)])
#	mqt.run(script='calico-wsrt-tens.py',job='cal_G_phase',section='S82_PHASE_CAL',config='tdlconf.profiles',args=['tiggerlsm.filename='+v.LSM,'ms_sel.msname='+v.MS,'ms_sel.ddid_index='+str(v.DDID),'cal_g_phase.g_phase.subtile_time=1'])	

def run():
	v.MS_List = glob.glob('*spw*.ms')
	#mergeMS('SB2-PTG477_cal.ms')
	per('MS',calMS)
	
run()

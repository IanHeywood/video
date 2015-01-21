from vidscp import *
import glob
import os

msName = glob.glob('sb*VIDEO*.ms')[0]
os.system('split-ms-spw.py '+msName)
spwlist = glob.glob('*spw*.ms')
for myms in spwlist:
	rflagMS(myms,5.0,5.0,'DATA')


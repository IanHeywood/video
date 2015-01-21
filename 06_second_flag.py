from vidscp import *
import glob
import os

spwlist = glob.glob('*spw*.ms')
for myms in spwlist:
	rflagMS(myms,5.0,5.0,'CORRECTED')


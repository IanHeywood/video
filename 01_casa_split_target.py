import glob
parentMS = glob.glob('../13B-308*.ms')[0]
tb.open(parentMS+'/FIELD')
names = tb.getcol('NAME')
for i in range(0,len(names)):
	name = names[i]
	if name.find('VIDEO') != -1:
		fieldid = name
tb.done()
opMS = parentMS.split('/')[-1].split('.')[1]+'_'+fieldid+'.ms'
split(vis=parentMS,outputvis=opMS,field=fieldid,datacolumn='corrected')

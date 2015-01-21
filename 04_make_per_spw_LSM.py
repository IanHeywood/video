from vidscp import *

msList = []

for i in range(15,16):
	msList += glob.glob('*spw'+str(i)+'.ms')

for myms in msList:
	spw = myms.split('_')[-1].replace('.ms','')
	imageList = glob.glob('*'+spw+'.ms*.fits')
#	pbList = glob.glob('*'+spw+'.*.pb.fits')
#	for pb in pbList:
#		imageList.remove(pb)
	if len(imageList) != 0:
		name_list = []
		for img in imageList:
			name_list.append(img.replace('.fits',''))
		info('Base Measurement Set '+myms)
		info('Considering the following files:')
		for item in name_list:
			info(spw+' '+item)			
		lsm_list = makeSkyModels(name_list)
		#lsm_list = remakeSkyModels(name_list)
		master_lsm = mergeLSMs(lsm_list)
	else:
		redinfo('No images found for '+str(spw)+' - cannot build sky model')

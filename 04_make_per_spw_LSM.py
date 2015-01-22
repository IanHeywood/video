from vidscp import *

for i in range(0,1):
	spw = 'spw'+str(i)
	imageList = glob.glob('*'+spw+'.ms*.fits')
	if len(imageList) != 0:
		name_list = []
		for img in imageList:
			xx = img.split('.')
			if 'alpha' in xx or 'flux' in xx:
				info('Ignoring '+img)
			else:
				name_list.append(img.replace('.fits',''))
		info('Considering the following files:')
		for item in name_list:
			info('    '+spw+' '+item)			
		lsm_list = makeSpectralSkyModels(name_list,7.0,3.0)
		#lsm_list = remakeSkyModels(name_list)
		master_lsm = mergeLSMs(lsm_list)
	else:
		redinfo('No images found for '+str(spw)+' - cannot build sky model')

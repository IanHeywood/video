# TO DO: You've got more than one invocation of the CASA clean task now so make ALL the variables global
import Pyxis
import Tigger
import std
import numpy
import os
import glob
import pyfits
from astLib import astWCS
from lofar import bdsm
from pyrap import tables
from astLib import astCoords as ac
from multiprocessing import Pool

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Global imaging parameters
im_niter = 1000
im_cell = '1.0arcsec'

im_npix_ref = [6000,6000]
im_cell_ref = '1.0arcsec'
im_niter_ref = 5000
im_robust_ref = 0.0

im_npix_out = [256,256]
im_cell_out = '1.0arcsec'
im_niter_out = 400
im_robust_out = 0.0

im_wplanes = 128
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::

def info(message):
	# Might want to add a log file or something
	print '\033[92m ----> '+message+'\033[0m'

def redinfo(message):
	# Might want to add a log file or something
	print '\033[91m ----> '+message+'\033[0m'

def imageMS(msName,imageName,npix,cell,niter,wplanes,robust,pcent):
	# General purpose imaging routine
	cc = 'clean('
	cc += 'vis="'+msName+'",'
	cc += 'imagename="'+imageName+'",'
	cc += 'mode="mfs",'
	cc += 'gridmode="widefield",'
	cc += 'wprojplanes='+str(wplanes)+','
	cc += 'niter='+str(niter)+','
	cc += 'imsize='+str(npix)+','
	cc += 'cell=["'+cell+'"],'
	cc += 'multiscale=[],'
	if pcent != '':
		cc += 'phasecenter = "'+pcent+'",'
	cc += 'weighting="briggs",'
	cc += 'robust=0.0)\n'
	cc += 'exportfits(imagename="'+imageName+'.image",fitsimage="'+imageName+'.fits",overwrite=True)\n'
	cc += 'exportfits(imagename="'+imageName+'.model",fitsimage="'+imageName+'.model.fits",overwrite=True)\n'
	cc += 'exportfits(imagename="'+imageName+'.flux",fitsimage="'+imageName+'.pb.fits",overwrite=True)'
	std.runcasapy(cc)
	deleteExtensions = ['.image','.flux','.residual','.model','.psf']
	for ext in deleteExtensions:
		syscall = 'rm -rf '+imageName+ext
		os.system(syscall)
		
def imageMSmfs(msName,imageName,npix,cell,niter,wplanes,robust,pcent):
	# image MFS nterms = 2
	cleanupimages = True
	cc = 'clean('
	cc += 'vis="'+msName+'",'
	cc += 'imagename="'+imageName+'",'
	cc += 'mode="mfs",'
	cc += 'nterms=2,'
	cc += 'gridmode="widefield",'
	cc += 'wprojplanes='+str(wplanes)+','
	cc += 'niter='+str(niter)+','
	cc += 'imsize='+str(npix)+','
	cc += 'cell=["'+cell+'"],'
	cc += 'multiscale=[],'
	if pcent != '':
		cc += 'phasecenter = "'+pcent+'",'
	cc += 'weighting="briggs",'
	cc += 'robust=0.0)\n'
	cc += 'exportfits(imagename="'+imageName+'.image.tt0",fitsimage="'+imageName+'.image.tt0.fits",overwrite=True)\n'
	cc += 'exportfits(imagename="'+imageName+'.flux",fitsimage="'+imageName+'.flux.fits",overwrite=True)\n'
	cc += 'exportfits(imagename="'+imageName+'.image.alpha",fitsimage="'+imageName+'.alpha.fits",overwrite=True)\n'
	cc += 'exportfits(imagename="'+imageName+'.image.alpha.error",fitsimage="'+imageName+'.alpha.error.fits",overwrite=True)\n'
	std.runcasapy(cc)
	if cleanupimages:
		deleteExtensions = ['.flux',
			'.image.alpha',
			'.image.alpha.error',
			'.image.tt0',
			'.image.tt1',
			'.model.tt0',
			'.model.tt1',
			'.psf.tt0',
			'.psf.tt1',
			'.residual.tt0',
			'.residual.tt1']
		for ext in deleteExtensions:
			syscall = 'rm -rf '+imageName+ext
			os.system(syscall)
		
def facetMS(msName,imageName,npix,cell,niter,N,F):
	print msName
	print imageName
	# Takes an image and cell size as per usual but divides the area up into
	# N*N facets each of which has a size of F*(npix/N)
	# Returns name_list which can be fed to makeFacetMontage to combine the images
	npix = npix[0] # Square images only for now
	msRA,msDec = getRefDirDegs(msName)
	# Spacing of facet centres
	dx =  int(npix / N)
	dy =  int(npix / N)
	xx,yy = numpy.mgrid[dx/2:N*dx:dx, dy/2:N*dy:dy]
	# Shift so zero is at the centre
	xc = npix / 2
	yc = npix / 2
	xx = xx - xc
	yy = yy - yc
	# Turn pixel grid into arcseconds 
	# as required by ac.shiftRADec
	cellAsec = float(cell.replace('arcsec',''))
	xxA = xx * cellAsec
	yyA = yy * cellAsec
	# Initialise lists
	name_list = []
	pc_list = []
	imsize_list = []
	# Populate lists
	counter = 0
	for i in range(0,xxA.shape[0]):
		for j in range(0,xxA.shape[1]):
			facet_ra,facet_dec = ac.shiftRADec(msRA,msDec,xxA[i][j],yyA[i][j])
			print facet_ra,facet_dec
			facet_ra_hms_parts = ac.decimal2hms(facet_ra,':').split(':')
			facet_dec_dms_parts = ac.decimal2dms(facet_dec,':').split(':')
			facet_ra_hms = facet_ra_hms_parts[0]+'h'+facet_ra_hms_parts[1]+'m'+facet_ra_hms_parts[2]+'s'
			facet_dec_dms = facet_dec_dms_parts[0]+'d'+facet_dec_dms_parts[1]+'m'+facet_dec_dms_parts[2]+'s'
			facet_pc = 'J2000,'+facet_ra_hms+','+facet_dec_dms
			# Make sure CASA clean doesn't choke if it doesn't like the imsize
			opt_npix_x = getOptimumSize(int(F*dx))
			opt_npix_y = getOptimumSize(int(F*dy))
			facet_npix = [opt_npix_x,opt_npix_y]
			facet_name = 'fctimgr_'+imageName+'_f'+str(counter)
			name_list.append(facet_name)
			pc_list.append(facet_pc)
			imsize_list.append(facet_npix)
			print facet_pc,facet_npix
			counter += 1
	# Run CASA imager
	cc = 'clean('
	cc += 'vis="'+msName+'",'
	cc += 'imagename='+str(name_list)+','
	cc += 'mode="mfs",'
	cc += 'gridmode="widefield",'
	cc += 'wprojplanes='+str(im_wplanes)+','
	cc += 'niter='+str(niter)+','
	cc += 'imsize='+str(imsize_list)+','
	cc += 'cell=["'+cell+'"],'
	cc += 'multiscale=[],'
	cc += 'phasecenter = '+str(pc_list)+','
	cc += 'weighting="briggs",'
	cc += 'robust=0.0)\n'
	# Export FITS images
	for item in name_list:
		cc += 'exportfits(imagename="'+item+'.image",fitsimage="'+item+'.fits",overwrite=True)\n'
		cc += 'exportfits(imagename="'+item+'.model",fitsimage="'+item+'.model.fits",overwrite=True)\n'
		cc += 'exportfits(imagename="'+item+'.flux",fitsimage="'+item+'.pb.fits",overwrite=True)\n'
	std.runcasapy(cc)
	deleteExtensions = ['.image','.flux','.residual','.model','.psf']
#	deleteExtensions = ['.flux','.residual','.model','.psf']
	for item in name_list:
		for ext in deleteExtensions:
			syscall = 'rm -rf '+item+ext
			os.system(syscall)
	return name_list
	
def fixMontageHeaders(infile,outfile):
	# Images produced by Montage do not have FREQ or STOKES axes
	# or information about the restoring beam. This confuses things like PyBDSM
	# infile provides the keywords to be written to outfile
	inphdu = pyfits.open(infile)
	inphdr = inphdu[0].header
	outhdu = pyfits.open(outfile,mode='update')
	outhdr = outhdu[0].header
	keywords = ['CTYPE','CRVAL','CDELT','CRPIX']
	for axis in [3,4]:
		for key in keywords:
			inkey = key+str(axis)
			outkey = key+str(axis)
			afterkey = key+str(axis-1)
			xx = inphdr[inkey]
			outhdr.set(outkey,xx,after=afterkey)
	outhdr.set('BUNIT',inphdr['BUNIT'],after=outkey)
	outhdr.set('BMAJ',inphdr['BMAJ'],after='BUNIT')
	outhdr.set('BMIN',inphdr['BMIN'],after='BMAJ')
	outhdr.set('BPA',inphdr['BPA'],after='BMIN')
	outhdu.flush()
	
def reprofits(proc):
	inpimg = proc[0]
	opimg = proc[1]
	hdr = proc[2]
	syscall = 'mProject '+inpimg+' '+opimg+' '+hdr
	info('Executing: '+syscall)
	os.system(syscall)

def makeFacetMontage(fits_list,final_image,opdir,ncpu,cleanup):
	# This really needs changing from a bunch of syscalls
	# to the Montage python API	
	opdir = opdir.replace('/','')
	os.system('mkdir '+opdir)
	os.system('mkdir '+opdir+'/repro')
	for item in fits_list:
		os.system('mv '+item+' '+opdir)
	os.system('mImgtbl '+opdir+' '+opdir+'/images.tbl')
	hdr = opdir+'/template.hdr'
	os.system('mMakeHdr '+opdir+'/images.tbl '+hdr)
	xx = glob.glob(opdir+'/*fits')
	proclist = []
	for item in xx:
		opimg = opdir+'/repro/'+item.split('/')[-1].replace('.fits','.repro.fits')
		if os.path.exists(opimg):
			info(opimg,'exits, skipping')
		else:
			proclist.append((item,opimg,hdr))
	pool = Pool(processes=ncpu)
	pool.map(reprofits,proclist)
	os.system('mImgtbl '+opdir+'/repro '+opdir+'/repro/images.tbl')
	os.system('mAdd -p '+opdir+'/repro '+opdir+'/repro/images.tbl '+opdir+'/template.hdr '+final_image+'.fits')
	if cleanup:
		os.system('rm -rf '+opdir)
	

def getOptimumSize(size):
	# Stolen from the CASA code
	'''
	This returns the next largest even composite of 2, 3, 5, 7
	'''
	def prime_factors(n, douniq=True):
		""" Return the prime factors of the given number. """
		factors = []
		lastresult = n
		sqlast=int(numpy.sqrt(n))+1
		# 1 pixel must a single dish user
		if n == 1:
			return [1]
		c=2
		while 1:
			if (lastresult == 1) or (c > sqlast):
				break
			sqlast=int(numpy.sqrt(lastresult))+1
			while 1:
				if(c > sqlast):
					c=lastresult
					break
				if lastresult % c == 0:
					break            
				c += 1
			factors.append(c)
			lastresult /= c
		if(factors==[]): factors=[n]
		return  numpy.unique(factors).tolist() if douniq else factors 
	n=size
	if(n%2 != 0):
		n+=1
	fac=prime_factors(n, False)
	for k in range(len(fac)):
		if(fac[k] > 7):
			val=fac[k]
			while(numpy.max(prime_factors(val)) > 7):
				val +=1
			fac[k]=val
	newlarge=numpy.product(fac)
	for k in range(n, newlarge, 2):
		if((numpy.max(prime_factors(k)) < 8)):
			return k
	return newlarge

def getCleanBeam(fitsfile):
	input_hdu = pyfits.open(fitsfile)[0]
	hdr = input_hdu.header
	bmaj = hdr.get('BMAJ')
	bmin = hdr.get('BMIN')
	bpa = hdr.get('BPA')
	return (bmaj,bmin,bpa)

def convolveImage(fitsfile,b):
	# Invoke the CASA ia.convolve2d tool to convolve an image
	# b is a gaussian (bmaj,bmin,bpa) tuple
	opimg = fitsfile.replace('.fits','.conv.img')
	opfits = fitsfile.replace('.fits','.conv.fits')
	cc = 'ia.open("'+fitsfile+'")\n'
	cc += 'ia.convolve2d('
	cc += 'outfile="'+opimg+'",'
	cc += 'type="gaussian",'
	cc += 'major="'+str(b[0])+'deg",'
	cc += 'minor="'+str(b[1])+'deg",'
	cc += 'pa="'+str(b[2])+'deg",'
	cc += 'overwrite=True)\n'
	cc += 'exportfits(imagename="'+opimg+'",fitsimage="'+opfits+'",overwrite=True)\n'
	std.runcasapy(cc)
	os.system('rm -rf '+opimg)
	return opfits

def getRefDirDegs(msName):
	# Get the direction of the MS in degrees
	# Assumes a single source only in the field table!
	tt = tables.table(msName.rstrip('/')+'/FIELD')
	refdir = tt.getcol('REFERENCE_DIR').squeeze()
	msRA = float(refdir[0])*180.0/numpy.pi
	msDec = float(refdir[1])*180.0/numpy.pi
	if msRA < 0.0:
		msRA += 360.0
	tt.done()
	return msRA,msDec

def deg2rad(a):
	return a*numpy.pi/180.0

def rad2deg(a):
	return a*180.0/numpy.pi
	
def angularSep(ra1,d1,ra2,d2):
	# Angular separation between two celestial positions
	ra1 = deg2rad(ra1)
	d1 = deg2rad(d1)
	ra2 = deg2rad(ra2)
	d2 = deg2rad(d2)
	sep = numpy.arccos((numpy.sin(d1)*numpy.sin(d2)) + (numpy.cos(d1)*numpy.cos(d2)*numpy.cos(ra1-ra2)))
	return sep*180.0/numpy.pi

def getOutlierFields(parentMS):
	# Search NVSS for potential confusing sources
	# See also: reprocess_NVSScat.py
	# Everything in degrees!
	catalogue = '../eastnvsssubset.txt' # search this catalogue, some cut of NVSS at present
	maxRadius = 5.0 # for sources within this radius
	outlierFlux = 0.1 # that are brighter than this many Jy <-- IMPLEMENT THIS
	halfImageExtent = float(im_cell_ref.rstrip('arcsec'))*(im_npix_ref[0]/2.0)/3600.0
	msRA,msDec = getRefDirDegs(parentMS)
	imageXrange = ((msRA - halfImageExtent),(msRA + halfImageExtent))
	imageYrange = ((msDec - halfImageExtent),(msDec + halfImageExtent))

	outliers = []
	catSources = []
	f = open(catalogue,'r')
	line = f.readline()
	while line:
		if line[0] != '#':
			cols = line.split()
			catSources.append((cols[0],float(cols[1]),float(cols[2]),float(cols[3])))
		line = f.readline()
	f.close()
	for src in catSources:
		radius = angularSep(msRA,msDec,src[1],src[2])
#		radius = (((msRA - src[1])**2.0) + ((msDec - src[2])**2.0))**0.5
		if radius < maxRadius: 
			if src[1] <= imageXrange[0] or src[1] >= imageXrange[1]:
				if src[2] <= imageYrange[0] or src[2] >= imageYrange[1]:
#			if radius < maxRadius and radius > minRadius:
					phCent = 'J2000 '+str(src[1])+'deg '+str(src[2])+'deg'
					outliers.append((src[0],phCent,round(src[3],2)))
	if len(outliers) > 0:
		info('Calibration will consider outlier fields:')
		for outlier in outliers:
			info(str(outlier))
	else:
		info('No outlier fields found.')
	return outliers

def tagOutliers(inputLSM):
	syscall = 'tigger-tag '+inputLSM+' +OUT -f'
	os.system(syscall)

def makeAllImages(msName,iteration,outliers,dryrun):
	# Image main lobe and facets where there are potential confusing sources
	# Set dryrun = True to just fill up and return the name_list, e.g. for source finding
	# on an existing set of images.
	refImage = 'ref_'+str(iteration)+'_'+msName
	size_list = [im_npix_ref]
	name_list = [refImage]
	pcent_list = ['']
	for outlier in outliers:
		imgName = 'out_'+str(iteration)+'_'+outlier[0]+'_'+msName
		size_list.append(im_npix_out)
		name_list.append(imgName)
		pcent_list.append(outlier[1])
	
	cc = 'clean('
	cc += 'vis="'+msName+'",'
	cc += 'imagename='+str(name_list)+','
	cc += 'mode="mfs",'
	cc += 'gridmode="widefield",'
	cc += 'wprojplanes='+str(im_wplanes)+','
	cc += 'niter='+str(im_niter)+','
	cc += 'imsize='+str(size_list)+','
	cc += 'cell=["'+im_cell+'"],'
	cc += 'multiscale=[],'
	cc += 'phasecenter = '+str(pcent_list)+','
	cc += 'weighting="briggs",'
	cc += 'robust=0.0)\n'
	
	for item in name_list:
		inp = item+'.image'
		out = item+'.fits'
		cc += 'exportfits(imagename="'+inp+'",fitsimage="'+out+'",overwrite=True)\n'
	
	if not dryrun:
		std.runcasapy(cc)
		deleteExtensions = ['.image','.flux','.residual','.model','.psf']
		for item in name_list:
			for ext in deleteExtensions:
				syscall = 'rm -rf '+item+ext
				os.system(syscall)
	
	return name_list

def makeSkyModels(name_list):
	# Run PyBDSM on a list of images
	# If the source finding is successful convert .gaul to Tigger format LSM
	# Sources in Tigger LSMs derived from outliers are given the OUT tag
	# in case selective subtraction is needed later.
	lsm_list = []
	for item in name_list:
		img = bdsm.process_image(item+'.fits',thresh_pix=7.0,thresh_isl=2.0)
		if not img.write_catalog(format='ascii',catalog_type='gaul'):
			redinfo('No sources found in '+item+'.fits')
		else:
			info('Wrote '+item+'.pybdsm.gaul')
#			b = getCleanBeam(item+'.fits')
#			minExtent = numpy.mean(b[0],b[1])*3600.0
#			info('Minimum extent for point/Gaussian distinction '+str(minExtent)+' arcsec')
#			tiggerConvert(item+'.pybdsm.gaul',minExtent)
			tiggerConvert(item+'.pybdsm.gaul')
			lsm_list.append(item+'.pybdsm.lsm.html')
			if item.find('outlier') != -1:
				syscall = 'tigger-tag '+item+'.pybdsm.lsm.html +OUT -f'
				os.system(syscall)
	return lsm_list
	
def remakeSkyModels(name_list):
	# Repeats the steps of makeSkyModels but without invoking PyBDSM
	# i.e. it reprocess the .gaul catalogues into Tigger format LSMs.
	# Useful for trying out various parameters during the conversion, but mainly\
	# tweaking the min-extent parameter.
	# MergeLSMs can then be run on the resulting lsm_list as normal.
	lsm_list = []
	for item in name_list: 
		if os.path.exists(item+'.pybdsm.gaul'):
			info('Converting '+item+'.pybdsm.gaul')
			b = getCleanBeam(item+'.fits')
			minExtent = numpy.mean(b[0],b[1])*3600.0
			info('Minimum extent for point/Gaussian distinction '+str(minExtent)+' arcsec')
			tiggerConvert(item+'.pybdsm.gaul',minExtent)
			lsm_list.append(item+'.pybdsm.lsm.html')
			if item.find('outlier') != -1:
				syscall = 'tigger-tag '+item+'.pybdsm.lsm.html +OUT -f'
				os.system(syscall)
		else:
			redinfo(item+'.pybdsm.gaul not found')
	return lsm_list
	
def mergeLSMs(lsmList):
	#lsmList = glob.glob('*spw0*.lsm.html')
	temp = []
	oplsm = 'master_'+lsmList[0].split('.ms')[0]+'.ms.pybdsm.lsm.html'
	for mylsm in lsmList:
		info('lsmlist: '+mylsm)
		lsm = Tigger.load(mylsm,verbose=True)
		temp.append(lsm)
	init = temp[0]
	for i in range(1,len(temp)):
		init.addSources(temp[i])
	Tigger.save(init,oplsm,verbose=True)
	return oplsm


def fitsInfo(fitsname = None):
	"""
	Get fits info
	"""
	hdu = pyfits.open(fitsname)
	hdr = hdu[0].header
	ra = hdr['CRVAL1']
	dra = abs(hdr['CDELT1'])
	raPix = hdr['CRPIX1']
	dec = hdr['CRVAL2']
	ddec = abs(hdr['CDELT2'])
	decPix = hdr['CRPIX2']
	try: freq0 = hdr['CRVAL3']
	except KeyError: freq0 = hdr['CRVAL3'] # casa image has RESTFRQ in header
	image = hdu[0].data
	wcs = astWCS.WCS(hdr,mode='pyfits')
	return {'image':image,'wcs':wcs,'ra':ra,'dec':dec,'dra':dra,'ddec':ddec,'raPix':raPix,'decPix':decPix,'freq0':freq0}

def sky2px(wcs,ra,dec,dra,ddec,cell):
	"""convert a sky region to pixel positions"""
	beam =  3.971344894833e-03 # beam size,
	dra = beam if dra<beam else dra # assume every source is at least as large as the psf
	ddec = beam if ddec<beam else ddec
	offsetDec = (ddec/2.)/cell
	offsetRA = (dra/2.)/cell
	raPix,decPix = wcs.wcs2pix(ra,dec)
	return np.array([int(raPix-offsetRA),int(raPix+offsetRA),int(decPix-offsetDec),int(decPix+offsetDec)])

def addSPI(fitsname_alpha=None, fitsname_alpha_error=None, lsmname=None, outfile=None):
	"""
		Add spectral index to a tigger lsm from a spectral index map (fits format)
		takes in a spectral index map, input lsm and output lsm name.
	"""
#	import pylab as plt
	print "INFO: Getting fits info from: %s, %s" %(fitsname_alpha, fitsname_alpha_error)

	fits_alpha = fitsInfo(fitsname_alpha)	# Get fits info
	fits_alpha_error = fitsInfo(fitsname_alpha_error)
	image_alpha = fits_alpha['image'][0,0] 	# get image data
	image_alpha_error = fits_alpha_error['image'][0,0]

	model = Tigger.load(lsmname)	# load sky model
	rad = lambda a: a*(180/np.pi) # convert radians to degrees
	psf = getCleanBeam(fitsname_alpha)
	beam = numpy.max((psf[0],psf[1]))
	print 'BEAM: ',beam,'!!!!!!!!!!!!!!!'

	for src in model.sources:
		ra = rad(src.pos.ra)
		dec = rad(src.pos.dec)
		tol = 30./3600. # Tolerance, only add SPIs to sources outside this tolerance (radial distance from centre)
		
		#beam = 3.971344894833e-03 # psf size, assume all sources are at least as large as the psf
		
		
		if np.sqrt((ra-fits_alpha["ra"])**2 + (dec-fits_alpha["dec"])**2)>tol: # exclude sources within {tol} of phase centre
			dra = rad(src.shape.ex) if src.shape  else beam # cater for point sources
			ddec = rad(src.shape.ex) if src.shape  else beam # assume source extent equal to the Gaussian major axis along both ra and dec axes
			rgn = sky2px(fits_alpha["wcs"],ra,dec,dra,ddec,fits_alpha["dra"]) # Determine region of interest

			#subIm_alpha = image_alpha[rgn[2]:rgn[3], rgn[0]:rgn[1]]  # Sample region of interest
			#subIm_alpha_error = image_alpha_error[rgn[2]:rgn[3], rgn[0]:rgn[1]]

			subIm_alpha_nonan = []
			subIm_alpha_error_nonan = []

			for (x,y) in zip(range(rgn[2],rgn[3]), range(rgn[0],rgn[1])):
				if np.isnan(image_alpha[x,y])==False and np.isnan(image_alpha_error[x,y])==False:
					subIm_alpha_nonan.append(image_alpha[x,y])
					subIm_alpha_error_nonan.append(image_alpha_error[x,y])

			#print "subIm_alpha_nonan      :", subIm_alpha_nonan
			#print "subIm_alpha_error_nonan:", subIm_alpha_error_nonan
			#print "length:

			#subIm_alpha_nonan = subIm_alpha[np.isnan(subIm_alpha_error)==False]
			#subIm_alpha_error = image_alpha_error[rgn[2]:rgn[3], rgn[0]:rgn[1]]
			#subIm_alpha_error_nonan = subIm_alpha_error[np.isnan(subIm_alpha_error)==False]

			subIm_weight = [1.0/subIm_alpha_error_nonan[i] for i in range(len(subIm_alpha_error_nonan))]
			#print "\n"
			#print "subIm_weight:", subIm_weight
			#print "\n"

			subIm_weighted = [subIm_alpha_nonan[i]*subIm_weight[i] for i in range(len(subIm_alpha_nonan))]

			#print "subIm_weighted:", subIm_weighted
			#print "\n\n"

			if len(subIm_weight)>0:
			#	subIm_normalization = subIm_weight.sum()
				subIm_normalization = np.sum(subIm_weight)
				#print "subIm__normalization:", subIm_normalization

			#if len(subIm_weighted)>0 and subIm_normalization>0:
			if len(subIm_weighted)>0:
			#	spi = subIm_weighted.sum()/subIm_normalization
				spi = np.sum(subIm_weighted)/subIm_normalization

				print "INFO: Adding spi: %.2g (at %.3g MHz) to source %s"%(spi,fits_alpha['freq0']/1e6,src.name)
				src.spectrum = Tigger.Models.ModelClasses.SpectralIndex(spi,fits_alpha["freq0"])
			else:
				print "ALERT: no spi info found in %s for source %s"%(fitsname_alpha,src.name)

	model.save(outfile)


def makeSpectralModel(lsm,alpha,beta):
	# USE SPHE'S VERSION
	# LSM is a Tigger format sky model
	# alpha is a FITS image of the alpha map from mfs nterms > 1
	# beta is the FITS image of the beta map from mfs nterms > 2
	input_hdu = pyfits.open(alpha)[0]
	hdr = input_hdu.header
	WCS = astWCS.WCS(hdr,mode='pyfits')
	alphas = getImageData(alpha)
	betas = getImageData(beta)
	mylsm = Tigger.load(lsm,verbose=True)
	sources = mylsm.sources
	padding = 1 # pixels either side of position, mean alpha and beta returned over this patch
	for src in sources:
		ra_d = rad2deg(src.pos.ra)
		dec_d = rad2deg(src.pos.dec)
		ra_pix,dec_pix = WCS.wcs2pix(ra_d,dec_d)
		if padding > 0:
			x0,x1 = int(ra_pix-padding),int(ra_pix+padding)
			y0,y1 = int(dec_pix-padding),int(dec_pix+padding)
			alpha = numpy.mean(alphas[y0:y1,x0:x1])
			beta = numpy.mean(betas[y0:y1,x0:x1])
			print x0,y0,x1,y1
		else:
			alpha = alphas[ra_pix,dec_pix]
			beta = betas[ra_pix,dec_pix]
			print ra_pix,dec_pix
		print src.name,ra_d,dec_d,alpha,beta
		src.spectrum = Tigger.Models.ModelClasses.SpectralIndex([alpha,beta],1400.0)
	Tigger.save(mylsm,'test.lsm.html',verbose=True)

#~ def mergeLSMs(inputList,outputLsm):
	# DEFUNCT VERSION USING .gaul FILES
	# DOES NOT PRESERVE TAGS!!
	#~ # Merge all the .gaul models into a master tigger format LSM
	#~ with open(outputLsm,'w') as outfile:
		#~ for inputFile in inputList:
			#~ with open(inputFile) as infile:
				#~ for line in infile:
					#~ outfile.write(line)
	#~ tiggerConvert(outputLsm)
	#~ info('Merged LSMs: '+str(inputList)+' into '+outputLsm+'.lsm.html')
	#~ return outputLsm.rstrip('.gaul')+'.lsm.html'

#def tiggerConvert(inputLsm,minExtent):
# Old version with wrong columns
#        tiggerLsm = inputLsm.replace('.gaul','.lsm.html')
#        syscall = 'tigger-convert '+inputLsm+' '+tiggerLsm+' -t ASCII --format "name Isl_id Source_id Wave_id ra_d E_RA de$
#        os.system(syscall)

def tiggerConvert(gaul):
	args = []
	tigger_convert  = x("tigger-convert")
	#Dictionary for establishing correspondence between parameter names in gaul files produced by pybdsm, and pyxis parameter names
	dict_gaul2lsm = {'Gaus_id':'name', 'Isl_id':'Isl_id', 'Source_id':'Source_id', 'Wave_id':'Wave_id', 'RA':'ra_d', 'E_RA':'E_RA', 'DEC':'dec_d', 'E_DEC':'E_DEC', 'Total_flux':'i', 'E_Total_flux':'E_Total_flux', 'Peak_flux':'Peak_flux', 'E_Peak_flux':'E_Peak_flux', 'Xposn':'Xposn', 'E_Xposn':'E_Xposn', 'Yposn':'Yposn', 'E_Yposn':'E_Yposn', 'Maj':'Maj', 'E_Maj':'E_Maj', 'Min':'Min', 'E_Min':'E_Min', 'PA':'PA', 'E_PA':'E_PA', 'Maj_img_plane':'Maj_img_plane', 'E_Maj_img_plane':'E_Maj_img_plane', 'Min_img_plane':'Min_img_plane', 'E_Min_img_plane':'E_Min_img_plane', 'PA_img_plane':'PA_img_plane', 'E_PA_img_plane':'E_PA_img_plane', 'DC_Maj':'emaj_d', 'E_DC_Maj':'E_DC_Maj', 'DC_Min':'emin_d', 'E_DC_Min':'E_DC_Min', 'DC_PA':'pa_d', 'E_DC_PA':'E_DC_PA', 'DC_Maj_img_plane':'DC_Maj_img_plane', 'E_DC_Maj_img_plane':'E_DC_Maj_img_plane', 'DC_Min_img_plane':'DC_Min_img_plane', 'E_DC_Min_img_plane':'E_DC_Min_img_plane', 'DC_PA_img_plane':'DC_PA_img_plane', 'E_DC_PA_img_plane':'E_DC_PA_img_plane', 'Isl_Total_flux':'Isl_Total_flux', 'E_Isl_Total_flux':'E_Isl_Total_flux', 'Isl_rms':'Isl_rms', 'Isl_mean':'Isl_mean', 'Resid_Isl_rms':'Resid_Isl_rms', 'Resid_Isl_mean':'Resid_Isl_mean', 'S_Code':'S_Code', 'Total_Q':'q', 'E_Total_Q':'E_Total_Q', 'Total_U':'u', 'E_Total_U':'E_Total_U', 'Total_V':'v', 'E_Total_V':'E_Total_V', 'Linear_Pol_frac':'Linear_Pol_frac', 'Elow_Linear_Pol_frac':'Elow_Linear_Pol_frac', 'Ehigh_Linear_Pol_frac':'Ehigh_Linear_Pol_frac', 'Circ_Pol_Frac':'Circ_Pol_Frac', 'Elow_Circ_Pol_Frac':'Elow_Circ_Pol_Frac', 'Ehigh_Circ_Pol_Frac':'Ehigh_Circ_Pol_Frac', 'Total_Pol_Frac':'Total_Pol_Frac', 'Elow_Total_Pol_Frac':'Elow_Total_Pol_Frac', 'Ehigh_Total_Pol_Frac':'Ehigh_Total_Pol_Frac', 'Linear_Pol_Ang':'Linear_Pol_Ang', 'E_Linear_Pol_Ang':'E_Linear_Pol_Ang'}

	#Dictionary for classifying a parameter as a general parameter or a polarization-specific parameter
	dict_pol_flag = {'Gaus_id':0, 'Isl_id':0, 'Source_id':0, 'Wave_id':0, 'RA':0, 'E_RA':0, 'DEC':0, 'E_DEC':0, 'Total_flux':0, 'E_Total_flux':0, 'Peak_flux':0, 'E_Peak_flux':0, 'Xposn':0, 'E_Xposn':0, 'Yposn':0, 'E_Yposn':0, 'Maj':0, 'E_Maj':0, 'Min':0, 'E_Min':0, 'PA':0, 'E_PA':0, 'Maj_img_plane':0, 'E_Maj_img_plane':0, 'Min_img_plane':0, 'E_Min_img_plane':0, 'PA_img_plane':0, 'E_PA_img_plane':0, 'DC_Maj':0, 'E_DC_Maj':0, 'DC_Min':0, 'E_DC_Min':0, 'DC_PA':0, 'E_DC_PA':0, 'DC_Maj_img_plane':0, 'E_DC_Maj_img_plane':0, 'DC_Min_img_plane':0, 'E_DC_Min_img_plane':0, 'DC_PA_img_plane':0, 'E_DC_PA_img_plane':0, 'Isl_Total_flux':0, 'E_Isl_Total_flux':0, 'Isl_rms':0, 'Isl_mean':0, 'Resid_Isl_rms':0, 'Resid_Isl_mean':0, 'S_Code':0, 'Total_Q':1, 'E_Total_Q':1, 'Total_U':1, 'E_Total_U':1, 'Total_V':1, 'E_Total_V':1, 'Linear_Pol_frac':1, 'Elow_Linear_Pol_frac':1, 'Ehigh_Linear_Pol_frac':1, 'Circ_Pol_Frac':1, 'Elow_Circ_Pol_Frac':1, 'Ehigh_Circ_Pol_Frac':1, 'Total_Pol_Frac':1, 'Elow_Total_Pol_Frac':1, 'Ehigh_Total_Pol_Frac':1, 'Linear_Pol_Ang':1, 'E_Linear_Pol_Ang':1}

	lines = [line.strip() for line in open(gaul)]

	for line in range(len(lines)):
		if lines[line]:
			if lines[line].split()[0] is not '#': 
				gaul_params = lines[line-1].split()[1:] #Parameter list is last line in gaul file that begins with a '#'
				break

	# Initialize lists for general and polarization parameters 
	lsm_params_general = []
	lsm_params_polarization = []

	for param in gaul_params:
		if dict_pol_flag[param] is 0:
			lsm_params_general.append(dict_gaul2lsm[param])
		if dict_pol_flag[param] is 1:
			lsm_params_polarization.append(dict_gaul2lsm[param])

	general_params_string = ' '.join(lsm_params_general)
	pol_params_string = ' '.join(lsm_params_polarization)

	output = gaul.replace('.gaul','.lsm.html')

	cluster = 30.0

	tigger_convert(gaul,output,"-t","ASCII","--format", general_params_string,
		"-f","--rename",
		"--cluster-dist",cluster,
	#	"--min-extent",MIN_EXTENT,
		split_args=False,
		*args);
	
def getSPW(name):
	parts = name.split('_')
	for part in parts:
		if part[0:3] == 'spw':
			spw = part.replace('spw','').replace('.ms','')
			spw = int(spw)
#			spw = int(part[3:])
	return spw

def emptyMS(msName):
	tt = tables.table(msName)
	if tt.nrows() == 0:
		empty = True
	else:
		empty = False
	return empty
	
def rflagMS(msName,timeDev,freqDev,datacolumn):
	cc = 'flagdata('
	cc+= 'vis="'+msName+'",'
	cc+= 'mode="rflag",'
	cc+= 'timedevscale='+str(timeDev)+','
	cc+= 'freqdevscale='+str(freqDev)+','
	cc+= 'datacolumn="'+datacolumn+'")'
	std.runcasapy(cc)

def splitMS(msName):
	os.system('split-ms-spw.py '+msName)

def pbcor(inpimg,pbimg,outimg,cutoff):
	# Open inpimg, divide it by pbimg where pbimg > cutoff
	# Write result to outimg
	# Shapes must match! Produce pbimg when you produce inpimg
	os.system('cp '+inpimg+' '+outimg)
	inphdu = pyfits.open(inpimg)[0]
	if len(inphdu.data.shape) == 2:
		inpdata = numpy.array(inphdu.data[:,:])
	elif len(inphdu.data.shape) == 3:
		inpdata = numpy.array(inphdu.data[0,:,:])
	else:
		inpdata = numpy.array(inphdu.data[0,0,:,:])
	pbhdu = pyfits.open(pbimg)[0]
	if len(inphdu.data.shape) == 2:
		pbdata = numpy.array(pbhdu.data[:,:])
	elif len(inphdu.data.shape) == 3:
		pbdata = numpy.array(pbhdu.data[0,:,:])
	else:
		pbdata = numpy.array(pbhdu.data[0,0,:,:])
	pbmask = pbdata < cutoff
	pbdata[pbmask] = numpy.nan
	pbcordata = inpdata/pbdata
	op = pyfits.open(outimg,mode='update')
	outhdu = op[0]
	if len(outhdu.data.shape) == 2:
		outhdu.data[:,:] = pbcordata
	elif len(outhdu.data.shape) == 3:
		outhdu.data[0,:,:] = pbcordata
	else:
		outhdu.data[0,0,:,:] = pbcordata
	op.flush()
	
def getImageData(fitsfile):
	inphdu = pyfits.open(fitsfile)[0]
	if len(inphdu.data.shape) == 2:
		inpdata = numpy.array(inphdu.data[:,:])
	elif len(inphdu.data.shape) == 3:
		inpdata = numpy.array(inphdu.data[0,:,:])
	else:
		inpdata = numpy.array(inphdu.data[0,0,:,:])
	return inpdata

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::

#~ msList = ['SB2-PTG672_spw0.ms']

#~ msList = glob.glob('*spw*.ms')

#~ for msName in msList:
	#~ outliers = getOutlierFields(msName)
	#~ name_list = makeAllImages(msName,0,outliers,False)
	#~ lsm_list = makeSkyModels(name_list)
	#~ master_lsm = mergeLSMs(lsm_list,'master_'+msName+'.lsm.html')



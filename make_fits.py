import os
import glob

to_find = ['.image.tt0','.image.alpha','.image.alpha.error','.image.beta']

for ext in to_find:
	infile = glob.glob('*'+ext)[0]
	outfile = infile+'.fits'
	if os.path.isfile(outfile):
		print outfile,'exists, skipping'
	else:
		exportfits(imagename=infile,fitsimage=outfile)
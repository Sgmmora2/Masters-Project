def LoadData():
	import numpy as np
	import pandas as pd
	import os as os
	import matplotlib.pyplot as pyplot
	import math as math
	os.chdir('Data')
	myarr = np.load('project data.npy')
	#df is the data in momentum eta phi form
	df = pd.DataFrame(data=myarr, columns=["label","weight","lep0pt","lep0eta","lep0phi",
										   "lep1pt","lep1eta","lep1phi",
										   "jet0pt","jet0eta","jet0phi",
										   "jet1pt","jet1eta","jet1phi",
										   "jet2pt","jet2eta","jet2phi",
										   "MET","METSig","NBJet","NEvent"])
	columns = df.columns.values
	#adding the x y z momentum data
	for i in range(5):
		p = df.iloc[:,2+3*i]*np.cosh(df.iloc[:,3+3*i])
		px = df.iloc[:,2+3*i]*np.cos(df.iloc[:,4+3*i])
		py = df.iloc[:,2+3*i]*np.sin(df.iloc[:,4+3*i])
		pz = df.iloc[:,2+3*i]*np.sinh(df.iloc[:,3+3*i])
		particlename = columns[2+3*i][:4]
		df[('%sp' % (particlename))] = p
		df[('%spx' % (particlename))] = px
		df['%spy' % (particlename)] = py
		df['%spz' % (particlename)] = pz    
	 #Linear momentum data has been added to the dataframe
     
	suffix4vc = ['p','px','py','pz']
	 
	#Arrays to allow me to reference properties of particles more systematically
	 
	for x in suffix4vc:
		df[('lepcombined%s') % (x)] = df[('lep0%s') % (x)] + df[('lep1%s') % (x)]
	#Assume E is equal to p
	df['lepcombinedinvariantmass'] = df['lepcombinedp'] - (df['lepcombinedpx']*df['lepcombinedpx']+df['lepcombinedpy']*df['lepcombinedpy']+df['lepcombinedpz']*df['lepcombinedpz']).apply(math.sqrt)

	for x in suffix4vc:
		df[('jetcombined%s') % (x)] = df[('jet0%s') % (x)] + df[('jet1%s') % (x)]
	df['jetcombinedinvariantmass'] = df['jetcombinedp'] - (df['jetcombinedpx']*df['jetcombinedpx']+df['jetcombinedpy']*df['jetcombinedpy']+df['jetcombinedpz']*df['jetcombinedpz']).apply(math.sqrt)

	for x in suffix4vc:
		df[('allcombined%s' % (x))] = df[('lep0%s') % (x)] + df[('lep1%s') % (x)] + df[('jet0%s') % (x)] + df[('jet1%s') % (x)]
	df['allcombinedinvariantmass'] = df['allcombinedp'] - (df['allcombinedpx']*df['allcombinedpx']+df['allcombinedpy']*df['allcombinedpy']+df['allcombinedpz']*df['allcombinedpz']).apply(math.sqrt)
	return df;
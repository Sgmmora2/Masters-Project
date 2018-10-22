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
for i in [0,1,2,3,4]:
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
    df[('jetcombined%s') % (x)] = df[('jet0%s') % (x)] + df[('jet1%s') % (x)]
    df[('allcombined%s' % (x))] = df[('lep0%s') % (x)] + df[('lep1%s') % (x)] + df[('jet0%s') % (x)] + df[('jet1%s') % (x)]
    
#Assume E is equal to p
def invMass(df,system,p,px,py,pz):
    return (df['%s%s' % (system,p)]*df['%s%s' % (system,p)] - df['%s%s' % (system,px)]*df['%s%s' % (system,px)] - df['%s%s' % (system,py)]*df['%s%s' % (system,py)] - df['%s%s' % (system,pz)]*df['%s%s' % (system,pz)]).apply(math.sqrt)

df['lepcombinedinvariantmass'] = invMass(df,'lepcombined','p','px','py','pz')
df['jetcombinedinvariantmass'] = invMass(df,'jetcombined','p','px','py','pz')
df['allcombinedinvariantmass'] = invMass(df,'allcombined','p','px','py','pz')

invariantmasses = [df.label, df.jetcombinedinvariantmass, df.lepcombinedinvariantmass, df.allcombinedinvariantmass]
invariantmassesdf = pd.DataFrame(data=invariantmasses, columns=['label','jet','lep','combined'])

signal = df[df.label >= 300000]
background = df[df.label <= 300000]

pyplot.figure(0)
pyplot.hist(signal.allcombinedinvariantmass, bins=200, weights = signal.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass All / Total Signal')
pyplot.savefig('testallsignal.png')
pyplot.figure(1)
pyplot.hist(signal.lepcombinedinvariantmass, bins=200, weights = signal.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass Jet / Total Signal')
pyplot.savefig('testJetsignal.png')
pyplot.figure(2)
pyplot.hist(signal.jetcombinedinvariantmass, bins=200, weights = signal.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass Lep / Total Signal')
pyplot.savefig('testLepsignal.png')

pyplot.figure(0)
pyplot.hist(background.allcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass All / Background')
pyplot.savefig('testallsigandback.png')
pyplot.figure(1)
pyplot.hist(background.lepcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass Lep /Background')
pyplot.savefig('testJetsigandback.png')
pyplot.figure(2)
pyplot.hist(background.jetcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass jet / Background')
pyplot.savefig('testLepsigandback.png')

pyplot.figure(3)
pyplot.hist(background.allcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass All / Signal and Background Superimposed')
pyplot.savefig('testallback.png')
pyplot.figure(4)
pyplot.hist(background.lepcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass lep / Signal and Background Superimposed')
pyplot.savefig('testJetback.png')
pyplot.figure(5)
pyplot.hist(background.jetcombinedinvariantmass, bins=200, weights = background.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass jet / Signal and Background Superimposed')
pyplot.savefig('testLepback.png')

pyplot.figure(6)
pyplot.hist(df.allcombinedinvariantmass, bins=200, weights = df.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass All / All events')
pyplot.savefig('testallfull.png')
pyplot.figure(7)
pyplot.hist(df.lepcombinedinvariantmass, bins=200, weights = df.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass lep / All events')
pyplot.savefig('testJetfull.png')
pyplot.figure(8)
pyplot.hist(df.jetcombinedinvariantmass, bins=200, weights = df.weight)
pyplot.ylabel('Frequency')
pyplot.xlabel('Combined Mass jet / All events')
pyplot.savefig('testLepfull.png')
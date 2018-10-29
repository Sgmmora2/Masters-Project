import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as pyplot
import math as math
import scipy.stats as sp

os.chdir('Data')
myarr = np.load('project data.npy')
#df is the data in momentum eta phi form
df = pd.DataFrame(data=myarr, columns=["label","weight","lep0pt","lep0eta","lep0phi",
										   "lep1pt","lep1eta","lep1phi",
										   "jet0pt","jet0eta","jet0phi",
										   "jet1pt","jet1eta","jet1phi",
										   "jet2pt","jet2eta","jet2phi",
										   "MET","METSig","NBJet","NEvent"])

#Add higgs mass dictionary

higgsdic = {
        307513:500,
        308496:130,
        308509:400,
        308513:300,
        308515:500,
        308517:200,
        308520:400,
        308527:700
        }

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

def histweighteddrac(data,weight):
    #ad hoc modification of the freedman diaconic formula for the weighted data to decide bin sizes
    constant = 2
    binsize = constant*2*sp.stats.iqr(data)/pow(len(data),1/4)
    binn = np.arange(min(data),max(data)+binsize,binsize)
    pyplot.hist(data,bins=binn, weights = weight, stacked = True, histtype = 'step')
def histweighteddracnostack(data,weight):
    #ad hoc modification of the freedman diaconic formula for the weighted data to decide bin sizes
    constant = 2
    binsize = constant*2*sp.stats.iqr(data)/pow(len(data),1/4)
    binn = np.arange(min(data),max(data)+binsize,binsize)
    pyplot.hist(data,bins=binn, weights = weight, stacked = False, histtype = 'step')

pyplot.figure(0)
histweighteddrac(signal.allcombinedinvariantmass, signal.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass All / Total Signal')
pyplot.savefig('testallsignal.png')
pyplot.figure(1)
histweighteddrac(signal.lepcombinedinvariantmass,signal.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass Jet / Total Signal')
pyplot.savefig('testJetsignal.png')
pyplot.figure(2)
histweighteddrac(signal.jetcombinedinvariantmass,signal.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass Lep / Total Signal')
pyplot.savefig('testLepsignal.png')

pyplot.figure(0)
histweighteddrac(background.allcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass All / SigandBack')
pyplot.savefig('testallsigandback.png')
pyplot.legend(('signal','background'))
pyplot.figure(1)
histweighteddrac(background.lepcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass Lep /SigandBack')
pyplot.savefig('testJetsigandback.png')
pyplot.legend(('signal','background'))
pyplot.figure(2)
histweighteddrac(background.jetcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass jet / SigandBack')
pyplot.legend(('signal','background'))
pyplot.savefig('testLepsigandback.png')

pyplot.figure(3)
histweighteddrac(background.allcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass All / Signal and Background Superimposed')
pyplot.savefig('testallback.png')
pyplot.figure(4)
histweighteddrac(background.lepcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass lep / Signal and Background Superimposed')
pyplot.savefig('testJetback.png')
pyplot.figure(5)
histweighteddrac(background.jetcombinedinvariantmass,background.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass jet / Signal and Background Superimposed')
pyplot.savefig('testLepback.png')

pyplot.figure(6)
histweighteddrac(df.allcombinedinvariantmass,df.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass All / All events')
pyplot.savefig('testallfull.png')
pyplot.figure(7)
histweighteddrac(df.lepcombinedinvariantmass,df.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass lep / All events')
pyplot.savefig('testJetfull.png')
pyplot.figure(8)
histweighteddrac(df.jetcombinedinvariantmass,df.weight)
pyplot.ylabel('N Events')
pyplot.xlabel('Combined Mass jet / All events')
pyplot.savefig('testLepfull.png')
histogramdata = pd.DataFrame(columns = ("Label","N"))
signalunfilted = signal
signal = signal[(signal.jet0pt*signal.jet0pt+
                signal.jet1pt*signal.jet1pt+
                signal.lep0pt*signal.lep0pt+
                signal.lep1pt*signal.lep1pt).apply(math.sqrt)/
                signal.allcombinedinvariantmass > 0.4]

for x,y in zip(signal.label.unique(),range(len(signal.label.unique()))):
    signalplot = signal[signal.label == x]
    
    if x in higgsdic:
        signalplot = signalplot[(signalplot.jetcombinedinvariantmass >= 0.85*higgsdic.get(x)-25) & (signalplot.jetcombinedinvariantmass <= higgsdic.get(x)+50)]
    
    pyplot.figure(9+4*y)
    histweighteddrac(signalplot.allcombinedinvariantmass,signalplot.weight)
    pyplot.ylabel('N Events')
    pyplot.xlabel('Mllbb / %s' % x)
    pyplot.savefig('%stestallfull.png' % x)
    
    
    pyplot.figure(10+4*y)
    histweighteddrac(signalplot.lepcombinedinvariantmass,signalplot.weight)
    pyplot.ylabel('N Events')
    pyplot.xlabel('Mll / %s' % x)
    pyplot.savefig('%stestlepfull.png' % x)
    
    
    
    pyplot.figure(11+4*y)
    histweighteddrac(signalplot.jetcombinedinvariantmass,signalplot.weight)
    pyplot.ylabel('N Events')
    pyplot.xlabel('Mbb / %s' % x)
    pyplot.savefig('%stestjetfull.png' % x)
    
    
    
    pyplot.figure(12+4*y)
    histweighteddracnostack(signalplot.jetcombinedinvariantmass,signalplot.weight)
    histweighteddracnostack(signalplot.lepcombinedinvariantmass,signalplot.weight)
    histweighteddracnostack(signalplot.allcombinedinvariantmass,signalplot.weight)
    
    pyplot.ylabel('N Events')
    pyplot.xlabel('Mbb/ll/llbb / %s' % x)
    pyplot.legend(('jet','lep','combined'))
    pyplot.savefig('%scombined.png' % x)
        
#From these plots the peaks on the jets and the combined peaks are quite clear and there's only one so some method of automatically obtaining peaks from the base data will be employed
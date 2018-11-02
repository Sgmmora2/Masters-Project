import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as pyplot
import math as math
import scipy.stats as sp

myarr = np.load('project data.npy')
#df is the data in momentum eta phi form
df = pd.DataFrame(data=myarr, columns=["label","weight","lep0pt","lep0eta","lep0phi",
										   "lep1pt","lep1eta","lep1phi",
										   "jet0pt","jet0eta","jet0phi",
										   "jet1pt","jet1eta","jet1phi",
										   "jet2pt","jet2eta","jet2phi",
										   "MET","METSig","NBJet","NEvent"])

#Original dataframe is preserved for debugging purposes
dfpro = df

#Notice that the system should be invariant to rotation around the z axis, the phi value of jet0 is forced to zero and other phi's are adjusted appropiately to reduce the dimensionality of the problem and eliminate a need
#for jet0phi in the learning

for i in ['lep0','lep1','jet0','jet1','jet2']:
    dfpro[('%sphi' % i)] = (dfpro[('%sphi' % i)]-dfpro.jet0phi) % (2*math.pi)

ProcessedNpy = dfpro.values   
np.save('Processed Data',ProcessedNpy)

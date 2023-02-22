#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 2022

@author: aennebrielmann
"""

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg # statistical tests

#%% ---------------------------------------------------------
# Specify directories; settings
#------------------------------------------------------------
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/'
resDir = dataDir + 'results/individuals/'
save = True

#%% ---------------------------------------------------------
# Load results; reshape
#------------------------------------------------------------
df = pd.read_csv(dataDir + '/perParticipantResults_cv.csv')

longDf = pd.wide_to_long(df,
                         stubnames=['med_rmse', 'avg_rmse'],
                         i=df.columns[:18],
                         j='model', sep='_', suffix='.*')
longDf = longDf.reset_index()
longDf['model'] = longDf['model'].str.replace('_results_','')
longDf['model'] = longDf['model'].str.replace('ure','')

#%% ---------------------------------------------------------
# descriptives
#------------------------------------------------------------
print(longDf.groupby('model')['med_rmse'].median())

fig = sns.stripplot(data=longDf, x='model', y='med_rmse',
                    alpha=.5, palette='tab20')
plt.legend(frameon=False)
fig.set_xticklabels(fig.get_xticklabels(), rotation=45,
                    horizontalalignment='right')
sns.despine()
plt.show()
plt.close()

#%% ---------------------------------------------------------
# ANOVA across participants
# -----------------------------------------------------------

# test assumptions
spher = pg.sphericity(dv='med_rmse', within='model', subject='subj', data=longDf)
normal = pg.normality(dv='med_rmse', group='model', data=longDf)

res = pg.rm_anova(dv='med_rmse', within='model', subject='subj', data=longDf,
                  detailed=True)
print(res)

postHocs = pg.pairwise_ttests(dv='med_rmse', within='model', subject='subj',
                              data=longDf, padjust='bonf')
postHocTable = postHocs.dropna()
printTable = postHocTable[['A', 'B', 'T', 'p-corr', 'BF10', 'hedges']]
print(printTable.to_latex(float_format="{:.3f}".format))

#%% ---------------------------------------------------------
# best model per participant
# -----------------------------------------------------------
peeps = []
bestRMSE = []
bestModel = []
for peep in longDf.subj.unique():
    thisDf = longDf[longDf.subj==peep]
    minRMSE = thisDf.med_rmse.min()
    bestRMSE.append(minRMSE)
    bestModel.extend(longDf[longDf.med_rmse==minRMSE].model.values)
    peeps.append(peep)
bestDf = pd.DataFrame({'subj': peeps, 'min_rmse': bestRMSE,
                       'best model': bestModel})

print(bestDf.groupby('best model')['min_rmse'].median())

fig = sns.stripplot(data=bestDf, x='best model', y='min_rmse', palette='tab20')
# add median rmse for LOO-average as reference
plt.hlines(0.203, 0, 8, 'k', 'dashed', label='leave-one-out-average')
plt.legend(frameon=False)
fig.set_xticklabels(fig.get_xticklabels(), rotation=45,
                    horizontalalignment='right')
sns.despine()
plt.show()
plt.close()

counts = bestDf['best model'].value_counts()
tableDf = bestDf.groupby(['best model']).median()
tableDf['N'] = counts
# tableDf.reset_index(inplace=True)
print(tableDf.to_latex(float_format="{:0.2f}".format))

#%% ---------------------------------------------------------
# attach best model per participant to .csv
# -----------------------------------------------------------
df['bestModel'] = bestDf['best model']
df.to_csv(dataDir + '/perParticipantResults_cv.csv')
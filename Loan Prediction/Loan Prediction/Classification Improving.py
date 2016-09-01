# General syntax to import specific functions in a library:
# from (library) import (specific library function)
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# General syntax to import a library but no functions: 
# import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define column names. Avoid trailing comma error
Cols = ["msg_id", "txndt", "orderno", "processor_id", "merchant_id", "bin", "brand", "acsProvider",
       "currency", "amt", "device", "os", "browser", "osVer", "browserVer", "txnOutcome", "txnStatus",
       "acsTimeSnap", "dsTimeSnap", "binClass", "binCountry", "bankName", "Trailing"]

# Load csv data into data frame
# Skip the header because column names are manually defined here
InputLocation = r'C:\Users\ycai\Documents\Data Files\Machine Learning Prototype\eaf_all_sample.txt'
data = pd.read_csv(InputLocation, names=Cols, skiprows=1)
print data

# Subset the data needed (bin and acsTimeSnap)
# Skip all 0 values
index = data.loc[:, 'acsTimeSnap'] > 0
df = data.loc[index, ['bin', 'acsTimeSnap']]
print df

# Generate summary statistics and specify the percentiles to include in the output
df_stats = df.groupby('bin').describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).unstack()

# Generate bin list and their unique bin classification
df_binclass = data.loc[:, ['bin', 'binClass']].groupby('bin').first()
print df_binclass

# Merge two data frames together
df_merged = df_stats.merge(df_binclass, left_index=True, right_index=True)
print df_merged

# Subset the merged data frame, left only traditional and riba, left only data with enough counts
df_merged_new = df_merged[df_merged.loc[:, 'binClass'].isin(['TRADITIONAL', 'RIBA'])]
df_merged_new = df_merged_new[df_merged_new.iloc[:, 0] > 1000]
print df_merged_new

# Perform KMeans Clustering on Mean and Std
kmeans = KMeans(init='k-means++', n_clusters=2)
df_factors = df_merged_new.iloc[:, [1, 2]]
kmeans.fit(df_factors)
print kmeans.labels_

# Add the kmeans column to the data frame
df_merged_kmeans = df_merged_new
df_merged_kmeans['Kmeans'] = kmeans.labels_
print df_merged_kmeans

# Output to the csv file
Output_Location = r'C:\Users\ycai\Documents\Data Files\BinClass Improvement\Stats.csv'
df_merged_kmeans.to_csv(Output_Location)
print "output complete"

# Visualization
h = 1000
x_min, x_max = df_factors.iloc[:, 0].min() + 1, df_factors.iloc[:, 0].max() - 1
y_min, y_max = df_factors.iloc[:, 1].min() + 1, df_factors.iloc[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(df_factors.iloc[:, 0], df_factors.iloc[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering of bin classification')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

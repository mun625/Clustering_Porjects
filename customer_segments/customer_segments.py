
# coding: utf-8

# # Creating Customer Segments

# In this project you, will analyze a dataset containing annual spending amounts for internal structure, to understand the variation in the different types of customers that a wholesale distributor interacts with.
# 
# Instructions:
# 
# - Run each code block below by pressing **Shift+Enter**, making sure to implement any steps marked with a TODO.
# - Answer each question in the space provided by editing the blocks labeled "Answer:".
# - When you are done, submit the completed notebook (.ipynb) with all code blocks executed, as well as a .pdf version (File > Download as).

# In[1]:

# Import libraries: NumPy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tell iPython to include plots inline in the notebook
# get_ipython().magic(u'matplotlib inline')

# Read dataset
data = pd.read_csv("wholesale-customers.csv")
print "Dataset has {} rows, {} columns".format(*data.shape)
print data.head()  # print the first 5 rows


# In[2]:

feature_list = list(data.columns)
feature_stat = pd.DataFrame(columns=['Minimum', 'Maximum', 'Mean_Value', 'Std_Dev'])

print "=============Basic Statistical data from features=============="
for feature in feature_list:
    feature_stat.loc[feature] = [np.min(data[feature]),np.max(data[feature]),np.mean(data[feature]),np.std(data[feature])]

print feature_stat


# ##Feature Transformation

# **1)** In this section you will be using PCA and ICA to start to understand the structure of the data. Before doing any computations, what do you think will show up in your computations? List one or two ideas for what might show up as the first PCA dimensions, or what type of vectors will show up as ICA dimensions.

# Answer: If we use PCA(n_componets = n), we will obtain n vectors. These vectors have two important properties. First these n vectors are orthogonal each other. It means these vectors are part of "basis of feature data's dimension". The data set's dimension can represent with some basis. These vectors are the 'n' most important basis among the basis.
# As you can see above, I calculate some basic statiscal data from features. In terms of Primary componont, (standard)deviation is very important because the bigger deviation means the more primary component. So I can predict that Fresh would be an first important feature for PCA. I mean the first basis vector of PCA is close to vector of Fresh. Similarly Grocery feature will impact a lot to second component of PCA.
# 
# ICA will return independent components set. We have 6 features now. But these are not (statistical) independent features. ICA algorithm will return (statistical) independent features. This means one feature doesn't affect other features value(statistically). For example, When we see original features, if someone buy fresh, then they are prone to buy milk. So the features in original data set is not independent each other. But ICA can find (hidden new) independent features of data set.
# 
# For example in this case, Many features can be a candidate of ICA components. Like Store size, Store type, location, or price, ... This is why we call them hidden layer. At first these factors are very hard to find, but usinc ICA we can find them and these are very meaningful.

# ###PCA

# In[3]:

# TODO: Apply PCA with the same number of dimensions as variables in the dataset
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(data)

# Print the components and the amount of variance in the data contained in each dimension
print pca.components_
print pca.explained_variance_ratio_

# Visualize the percent of variance explained
x = np.arange(7)
plt.plot(x, np.cumsum(np.insert(pca.explained_variance_ratio_, 0, 0)), '-')
plt.show()


# **2)** How quickly does the variance drop off by dimension? If you were to use PCA on this dataset, how many dimensions would you choose for your analysis? Why?

# Answer: The value drops really quickly as components go backward. Particularly, the first two components shows large percent of variance explained by that dimension. From third elements, we can see the variance drops dramatically. So based on elbow method, I will choose two components as new eigenvectors. So our new frame would have two dimensions.

# **3)** What do the dimensions seem to represent? How can you use this information?

# Answer: Let's talk about two major basis vectors. First one is [-0.97653685 -0.12118407 -0.06154039 -0.15236462  0.00705417 -0.06810471]. As we can see this one is almost similar to firstfeature(Fresh). So this information says that fresh_feature will be a important factor. Second major component is [-0.11061386  0.51580216  0.76460638 -0.01872345  0.36535076  0.05707921]. From this vector, we can check that the weighted sum of milk, grocery and detegernt_paper would be a major factor. we call these new componets 'eigenvectors'. Eigenvectors have given us a much more useful axis to frame the data in. So eigenvectors can be thought as a new imortant features. When we see second eigenvector, we can check that milk, grocery, and detergent got lumped together. It means they are tend to move together, and this amount is important to divide types of customers.
# 
# So How can we use this information? we can use it to divide customer segmentations. For simple example we can divide four groups according to their consumption of 'fresh' and 'milk,grocery,detergent'. Let's say A, B, C, D groups. So people in group A rarely buy Fresh, but they really like to buy milk. We can select two different people in A, and we can perform A/B testing! For example providing different type of delivery services to them, we can check which service is better.

# ###ICA

# In[5]:

# TODO: Fit an ICA model to the data
# Note: Adjust the data to have center at the origin first!
from sklearn.decomposition import FastICA
ica = FastICA(n_components=6, random_state = 5)
ica.fit(data-np.mean(data))

# Print the independent components
print ica.components_
# Visualize the independent components
pd.DataFrame(ica.components_, columns = data.columns).plot(kind = 'bar', figsize = (12,6))


# **4)** For each vector in the ICA decomposition, write a sentence or two explaining what sort of object or property it corresponds to. What could these components be used for?

# Answer: Above vectors are ICA components. As you can see, second and sixth vector is mostly consisted of delicatessen and grocery products respectively. So highest value with second vector means they are like to buy delicatessen products. There is more interesting things at third and fourth components. Third vector is mostly explained by big negative and positive values of detergents and grocery features. This means there is anti-corelationship between detergents and grocery products. Moreover customer with high value of thrid components are tend to buy grocery foods, low value of third components are ten to buy detergents. In other words, people who like to buy grocery food normally don't like to buy detergents and vice versa.(Anti-corelationship)
# We can apply same logic to fourth vector which are chiefly consisted of frozen and delicatessen. So we can check that frozen and delicatessen features are in a anti-corelationship.
# 
# Also the components are (statistically) independent each other. So regardless of second vectors value(mostly explained by delicatessen feature), we can say that there is anti-correlationship between frozen and delicatessen).

# ##Clustering
# 
# In this section you will choose either K Means clustering or Gaussian Mixed Models clustering, which implements expectation-maximization. Then you will sample elements from the clusters to understand their significance.

# ###Choose a Cluster Type
# 
# **5)** What are the advantages of using K Means clustering or Gaussian Mixture Models?

# Answer: First let's talk about K means algorithm.
# - (Pros) Simple and fast for data with low dimensionality.
# - (Cons) K means cannot discern outliers.
# 
# Follwing properties are Gaussian EM models'.
# - (Pros) Soft clustering is enabled. (calculate probablity of belonging to each group)
# - (Pros) Obtain a density estimation for each cluster (also can discern outliers)
# - (Cons) With a large data set, calculation can be slow.

# **6)** Below is some starter code to help you visualize some cluster data. The visualization is based on [this demo](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html) from the sklearn documentation.

# In[6]:

# Import clustering modules
from sklearn.cluster import KMeans
from sklearn.mixture import GMM


# In[7]:

# TODO: First we reduce the data to two dimensions using PCA to capture variation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)

reduced_data = pca.fit_transform(data)
print reduced_data[:10]  # print upto 10 elements


# In[8]:

# TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
# The visualizer below assumes your clustering object is named 'clusters'
# implemnt two algorithm (KMean & GMM) so that we can compare two algorithms' results.
gmm_clusters_7 = GMM(n_components=7)
gmm_clusters = GMM(n_components=6)
gmm_clusters_7.fit(reduced_data)
gmm_clusters.fit(reduced_data)

km_clusters_7 = KMeans(n_clusters=7)
km_clusters = KMeans(n_clusters=6)
km_clusters_7.fit(reduced_data)
km_clusters.fit(reduced_data)
print gmm_clusters


# In[9]:

# Plot the decision boundary by building a mesh grid to populate a graph.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
hx = (x_max-x_min)/1000.
hy = (y_max-y_min)/1000.
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

# Obtain labels for each point in mesh. Use last trained model.
gmm_Z_7 = gmm_clusters_7.predict(np.c_[xx.ravel(), yy.ravel()])
gmm_Z = gmm_clusters.predict(np.c_[xx.ravel(), yy.ravel()])
km_Z_7 = km_clusters_7.predict(np.c_[xx.ravel(), yy.ravel()])
km_Z = km_clusters.predict(np.c_[xx.ravel(), yy.ravel()])


# In[10]:

# TODO: Find the centroids for KMeans or the cluster means for GMM 
gmm_centroids_7 = gmm_clusters_7.means_
gmm_centroids = gmm_clusters.means_
km_centroids_7 = km_clusters_7.cluster_centers_
km_centroids = km_clusters.cluster_centers_


# In[26]:

# Put the result into a color plot
gmm_Z_7 = gmm_Z_7.reshape(xx.shape)
gmm_Z = gmm_Z.reshape(xx.shape)
km_Z_7 = km_Z_7.reshape(xx.shape)
km_Z = km_Z.reshape(xx.shape)

# plot for GMM algorithm
print "-------------Customer_segmentaion by GMM algorithm---------------"
plt.figure(1)
plt.clf()
plt.imshow(gmm_Z_7, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(gmm_centroids_7[:, 0], gmm_centroids_7[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

plt.figure(1)
plt.clf()
plt.imshow(gmm_Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(gmm_centroids[:, 0], gmm_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color=['w','b','g','y','r','c'], zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# analyze the centroids coordinate
print pca.inverse_transform(gmm_centroids)
gmm_center_data = pd.DataFrame(pca.inverse_transform(gmm_centroids), columns=data.columns)
plt.figure(1)
plt.clf()
gmm_center_data.plot(kind = 'bar')
plt.show()

print "--------------------------------------------------------------------"

# plot for KMeans algorithm
print "-------------Customer_segmentaion by KMeans algorithm---------------"
plt.figure(1)
plt.clf()
plt.imshow(km_Z_7, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(km_centroids_7[:, 0], km_centroids_7[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

plt.figure(1)
plt.clf()
plt.imshow(km_Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(km_centroids[:, 0], km_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# analyze the centroids coordinate
print pca.inverse_transform(km_centroids)
km_center_data = pd.DataFrame(pca.inverse_transform(km_centroids), columns=data.columns)
plt.figure(1)
plt.clf()
km_center_data.plot(kind = 'bar')
plt.show()


# #### **6)** Explanation about above models
# 
# #### 1. 6-clusters V.S. 7-clusters
# Let's compare 6-cluster model and 7-cluster model. First picture is describing the model with 7-clusters. Second picture is describing the 6-clusters model. There is some reason why I choose 6-clusters model instead of 7's. In 7-cluster model, there is a redundant group. (First picture)when you see left-below area of picture, the purple group contains only one component. It's very useless and waste of resources in terms of segmentation. Instead, we can classify that one component to another group and reduce the group size so that we can manage more easily.
# 
# #### 2. Gaussian Mixture Models V.S. K Means
# From now on, let's compare 6-cluster GMM model and 6-cluster KM model which are second and fourth pictures. I would like to say Gaussian models is more suitable in this case. Becuase KM algorighm seprate according to distance from centroids. So they just geometrically divide the points. In our case they judge value is high or not or medium and separate. I think it's not a good segmentation. Instead, GMM considers probability (tendency of group). This is why their groups' shape seems interesting. At K-means, segmentation shape is similar to rectangular shape. Instead, GMM's shape is more sharper or wider. This is because divided by the tendency not by only geometrical position.
# 
# #### 3. Explanation about inverse transforms
# In the middle of code, I insert "pca.inverse_transform(gmm_centroids)", "pca.inverse_transform(km_centroids)" and visualize the centorids respectively. This is restoration method that unzip a compressed data. So using this method I can obtain original features(frozen, grocery, ...). 
# To check easily (which components represent centroids), I visualize it by coloring. I used color ['white','blue','green','yellow','red','cyon']. So First value (in bar graph) represents white centroid, second value (in bar graph) represents blue centroid, ....

# **7)** What are the central objects in each cluster? Describe them as customers.

# Answer: These central points represent each clusters. I divided to 6 groups. I can say that two people who are belonging to same group have similar pattern of consumption. And also they will respond very similarly to some events.
# Furthermore each group can be represented by their centroid. Using inverse_transform, I printed the features' information of centroids.
# 
# As you can see above bar-graph(gmm), You can visually check the properties of each group. For example group 5 is heavy consumer and they like to buy grocery products. Group 6 usually buy fresh a lot. Group 0 and Group 2 normally spend not too much. Likewise we can analyze the centroid data and this would represent each segmentation of customers.

# ###Conclusions
# 
# ** 8)** Which of these techniques did you feel gave you the most insight into the data?

# Answer: As you can see on second picture I tried to classify using KMeans algorithm. But the segmentation seems not that meaningful. I believe that the segmentation with GMM classifier seems more meaningful. Gaussian Mixture Model. It considers not only distance but also probability. So the GMM is more natural separation. On the other hands, KMeans algorithm's separation looks not meaningful. It just divide the area geometricaly.
# 
# Let's talk about PCA and ICA. Basically ICA is not used for reducing dimensionality. It is userful when separate mixed signals. If we want to discern featured signal from others, we can use ICA. On the other hand, PCA is the optimized method to reduce dimensionality or to compress the data size. So in this case I would like to choose PCA method so that compress data and reduce dimensionality which can prevent the overfitting problem.

# **9)** How would you use that technique to help the company design new experiments?

# Answer: Customer segmentation is really important task to marketing part. Using this data we can apply suitable advertisement or promotion to each segmentation group. For example, for green group customers, we can give them to sale promotion code or coupon of fresh food. Or delivery service for another group.
# 
# Also we can manage group respectively. If we manage whole people at once, there are many problems. Group with large number(like light green, brown group)'s tendency is much more powerfull than small groups tendency. So, in this case people of group(light green)'s dissatisfaction can be ignored by noise of major group. However using a segmentation and managing respectively can prevent these kinds of problems.
# 
# Moreover, once we classify the data set, A/B testing become really useful. When we test something, there are really lots of variables. And it is very important to manage variables. Execpt target variables, we should keep other variables same values. As we classify the data set to some groups, we can perform the A/B test on same group. It will maintain other feautres same and only variable related to testing would be different. It is ideal environment to A/B testing.

# **10)** How would you use that data to help you predict future customer needs?

# Answer: I will store data separately based on their group. I can take a suvey from people who belongs to different groups. And their opinion can represent their group. So I can predict their needs likewise.
# 
# People in same group are prone to behave similar. So when we apply machine learning, training should not be weighted on one group. I mean, when we split the training data we can choose data uniformly in each sections.

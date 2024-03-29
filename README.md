# SkClone
## A project to build Machine Learning models from scratch in python
## Author: Suprabhat Rijal
### Overview:
SkClone is a project to build a machine learning library that is able to solve ML problems like regression and classification. It is structured similarly to a popular open-source ML library "**SkLearn**",  hence the name "**SkClone**". At this point, this is just a hobby project to figure out how various machine learning models work.
### What I've implemented so far:
1) **Linear Regression:** This implementation utilizes gradient descent(not stochastic)  to locate the minima and updates the weight and bias to fit the line. An example of linear regression using SkClone is plotted below: <img src="https://i.imgur.com/S31lhjp.png"/> 
2) **K-Nearest-Neighbors Classification:** This implementation simply finds a specified number(denoted by "K") of closest datapoints(Neighbors) and holds a majority vote between them in orders to classify new datapoints. An example of K-Nearest-Neighbors classification using SkClone is plotted below:<img src="https://i.imgur.com/WyBLf8W.png"/>
3) **K-Means Classification:** This can be used to categorize uncategorized data into a specified number of categories(denoted by "K"). The model crashes at times due to improper intialization of the centroid which will be fixed after the implementation of "**KMean++**". An example of K-Nearest-Neighbors classification using SkClone is plotted below:<img src="https://i.imgur.com/KDqza4d.png"/> 
### Next Few Milestones:
1) Implement the "**K-Means++**" algorithm to properly initialize the centroids in my implementation of the **K-Means** algorithm
2) Write "**setup.py**" so that people can download the package using pip. 





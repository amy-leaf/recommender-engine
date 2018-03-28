# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:25:57 2017

@author: ABDULRAHMAN
"""

"""
http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as op

import numpy as np

#dpoints = np.array([['user 1', 'Drama', 9.97],
#           ['user 1', 'Action', 27.31],
#           ['user 1', 'Comedy', 5.77],
#           ['user 2', 'Drama', 5.55],
#           ['user 2', 'Action', 37.74],
#           ['user 2', 'Comedy', 5.77],
#           ['user 3', 'Drama', 10.32],
#           ['user 3', 'Action', 31.46],
#           ['user 3', 'Comedy', 18.16]])

#fig = plt.figure()
#ax = fig.add_subplot(111)

def barplot(ax, dpoints):
    '''
    Create a barchart for data across different genres with
    multiple users for each genre.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    
    # Aggregate the users and the genres according to their
    # mean values
    users = [(user, np.mean(dpoints[dpoints[:,0] == user][:,2].astype(float))) 
                  for user in np.unique(dpoints[:,0])]
    
    genres = [(genre, np.mean(dpoints[dpoints[:,1] == genre][:,2].astype(float))) 
                  for genre in np.unique(dpoints[:,1])]
    
    # sort the users, genres and data so that the bars in
    # the plot will be ordered by genre and user
    users = [user[0] for user in sorted(users, key=op.itemgetter(1))]
    genres = [genre[0] for genre in sorted(genres, key=op.itemgetter(1))]
    
    dpoints = np.array(sorted(dpoints, key=lambda x: genres.index(x[1])))
#    print(dpoints)
    # the space between each set of bars
    space = 0.3
    n = len(users)
    width = (1 - space) / (len(users))
    # Create a set of bars at each position
    for i,user in enumerate(users):
        indeces = range(1, len(genres)+1)
        vals = dpoints[dpoints[:,0] == user][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=user, 
               color=cm.Accent(float(i) / n))
    
    # Set the x-axis tick labels to be equal to the genres
    ax.set_xticks(indeces)
    ax.set_xticklabels(genres)
    plt.setp(plt.xticks()[1], rotation=90)
    
    # Add the axis labels
    ax.set_ylabel("Movies (%)")
    ax.set_xlabel("Genres")
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
        
#barplot(ax, dpoints)
#savefig('barchart_3.png')
#plt.show()
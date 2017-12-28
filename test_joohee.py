# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:25:40 2017

@author: Joohee Lee
"""

xx = test_images[1111]
xx = np.stack([xx,xx])
xx = xx.T
xx = xx[:,1:].T

xx = test_images[13]
xx = np.stack([xx,xx])
xx = xx.T
xx = xx[:,1:].T

y_score = sess.run(y, feed_dict={x: xx, keep_prob:1})
y_score = y_score.T
y_score_= np.zeros([7,])
for i in range(y_score.shape[0]):
    y_score_[i]= y_score[i]
    
show(test_images[1999])
x_1 = np.arange(7)
plt.bar(x_1, height = y_score_)
plt.xticks(x_1+.5, ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])


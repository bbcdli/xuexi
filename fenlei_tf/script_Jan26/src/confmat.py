import numpy as np
import matplotlib.pyplot as plt


# def Five2Two(conf_mat):



a=[[30,1,0,0,1],[0,22,1,2,4],[6,1,36,1,0],[0,3,2,16,0],[2,3,2,0,66]]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

res = ax.imshow(np.array(a), cmap=plt.cm.gray, interpolation='nearest')

#plt.xticks(np.arange(0,5), ['Empty', 'Kiwa', 'Person', 'Person+KiWa', 'Wheelchair'])
#plt.yticks(np.arange(0,5), ['Empty', 'Kiwa', 'Person', 'Person+KiWa', 'Wheelchair'])
plt.xticks(np.arange(0,6), ['hinten', 'links', 'oben', 'rechts', 'unten','vorn']) #hy
plt.yticks(np.arange(0,6), ['hinten', 'links', 'oben', 'rechts', 'unten','vorn']) #hy


plt.show()

for i in a:
    print i




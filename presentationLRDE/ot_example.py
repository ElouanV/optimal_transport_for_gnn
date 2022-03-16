import numpy as np
import matplotlib.pylab as plt
import ot
'''

######## COLOR TRANSFERT ########
r = np.random.RandomState(42)

def im2mat(img):
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    return X.reshape(shape)


T1= plt.imread('sunset.jpg').astype(np.float64)/256
T2= plt.imread('mountain.jpg').astype(np.float64)/256
print(T1[0,0,:])

day = im2mat(T1)
sunset = im2mat(T2)


nb = 1000
idx1 = r.randint(day.shape[0], size=(nb,))
idx2 = r.randint(sunset.shape[0], size=(nb,))

Xs = day[idx1, :]
Xt = sunset[idx2, :]


plt.subplot(1,2,1)
plt.scatter(Xs[:,0], Xs[:,2], c=Xs)
#plt.axis([0,1,0,1])
plt.xlabel('Red')
plt.ylabel('Blue')
plt.xticks([])
plt.yticks([])
plt.title('Pink sky')

plt.subplot(1, 2, 2)

plt.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
#plt.axis([0, 1, 0, 1])
plt.xlabel('Red')
plt.ylabel('Blue')
plt.title('Mountain')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('graphs')
plt.show()

ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)

transp_Xt_emd = ot_emd.inverse_transform(Xt=sunset)

I2t = mat2im(transp_Xt_emd, T2.shape)

plt.figure()
plt.imshow(I2t)
plt.axis('off')
plt.title('Color transfer')
plt.tight_layout()
plt.savefig('color_transfer.jpg')
plt.show()

'''

##########################################

from poissonblending import blend
img_mask = plt.imread('masqlecoq.jpg')
img_mask = img_mask[:,:,:3] # remove alpha

img_source = plt.imread('clairescaled.jpg')
img_source = img_source[:,:,:3] # remove alpha

img_target = plt.imread('monalisa.jpg')
img_target = img_target[:,:,:3] # remove alpha

nbsample = 500
off = (35,-15)
seamless_copy = blend(img_target, img_source, img_mask, reg=5, eta=1, nbsubsample=nbsample, offset=off, adapt='kernel')

plt.figure()
plt.imshow(seamless_copy)
plt.axis('off')
plt.savefig('pics/fusion7')
plt.show()
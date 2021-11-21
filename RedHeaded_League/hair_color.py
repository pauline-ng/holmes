#!/usr/bin/env python


from PIL import Image
import glob
import os
import numpy as np
from skimage import color
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt 
#import cv2


folders = ["pics/blonde/blonde*jpg", "pics/brunette/brown*jpg", "pics/redheads/red*jpg"]
X =[]
Y = []
imgs_with_channel = []

def resize_images ():
    jpg_files=glob.glob (folders[0])
    for jpg_file in jpg_files:
        min_width = 1000
        min_height = 1000

        im = Image.open(jpg_file) # Can be many different formats.
        pix = im.load()
        print (im.size)  # Get the width and hight of the image for iterating over
        if im.size[0] < min_width:
            min_width = im.size[0]
        if im.size[1] < min_height:
            min_height = im.size[1]

        
print ("hello")

# assign jpg_files all of the hair-colored jpgs
jpg_files = []
for folder in folders:
    jpg_files.extend (glob.glob (folder))

training_data = []
IMG_SIZE = 50

for jpg_file in jpg_files:
    label = os.path.basename (jpg_file).replace (".jpg", "")
    label = ''.join([i for i in label if not i.isdigit()]) 

    im = Image.open(jpg_file) # Can be many different formats.

    x=20
    y=20
    IMG_PX_SIZE = 20 # pixels
    rgb_img = io.imread (jpg_file)
    
    # resize images to all the same size
    #rgb_img_resize = rgb_img.resize(IMG_SIZE, IMG_SIZE, 0)
    #rgb_img = rgb_img.resize (20,20,0)
    rgb_img = resize (rgb_img, (IMG_PX_SIZE,IMG_PX_SIZE))
    # get HSV representation of image
 #   hsv_img = rgb2hsv (rgb_img)
    
    # get Hue values of image (igeore saturation and value)
    # do over, need to resize
  #  hue_img = hsv_img[0:x, 0:y, 0]
    #hue_img_resize = cv2
 #   print (hue_img.shape)
    
    #print (pix[x,y])  # Get the RGBA Value of the a pixel of an image
  #  print (hue_img[x,y])
   # print (rgb_img.reshape (1, IMG_PX_SIZE*IMG_PX_SIZE*3)[0])  # flatten into an array
 #   X.append (hue_img.reshape ((hue_img.shape[0], x*y*3)))
    #X.append (np.array(hue_img))    
    X.append (rgb_img.reshape (1, x*y*3)[0]) # flatten into a 1-D array
    Y.append (label)
    

 #   img_with_channel = np.expand_dims (hue_img, axis=-1)
  #  imgs_with_channel.append (img_with_channel)
#        print (depth_with_channel.shape)


    # final step-forming the training data list with numpy array of the images 
   # training_data.append([np.array(hue_img), np.array(label)]) 
  


# In[22]:


#IMG_SIZE = 50

#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import LabelEncoder




# In[23]:


num_groups = 3  # blonde, red, brunette

model = KNeighborsClassifier(n_neighbors=num_groups)

model.fit (X, Y)


# prediction = model.predict (X)

# In[24]:


print (model.predict (X))


# In[19]:


#print "generated examples to predict:\n",predict,"\n"
# predict class probabilities for each class for each value and convert to DataFrame
probs = model.predict_proba(X)
#print ("all probabilities:\n", probs, "\n")
for index, prob in enumerate (probs):
    jpg_file = jpg_files[index]
    print (jpg_file)
    print (str(prob))


# red1 has a component of blonde in it, not truly red
# 

# In[20]:



# visualization
# https://stackoverflow.com/questions/65269382/how-can-i-visualize-the-test-samples-of-k-nearest-neighbour-classifier

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, Y, clf=model, legend=2)# Adding axes annotations
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Knn with K=3")
plt.show()


def get_neighbors(xs, sample):
    neighbors = [(x, np.sum(np.abs(x - sample))) for x in xs]
    neighbors = sorted(neighbors, key=lambda x: x[1])
    return np.array([x for x, _ in neighbors])

colors_array = []
for hair_color in Y:
    color = hair_color
    if hair_color == "blonde":
        color = "yellow"
    colors_array.append (color)
    
_, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
for i in range(4):
    sample = X[i]
    neighbors = get_neighbors(X, sample)
 #   print (X)
#    print (X[:,0])
 #   ax[i].scatter(X[:, 0], X[:, 1], c="skyblue")
    ax[i].scatter(neighbors[:, 0], neighbors[:, 1], c=colors_array) #edgecolor="green")
    ax[i].scatter(sample[0], sample[1], marker="+", c="red", s=100)
    ax[i].set(xlim=(-2, 2), ylim=(-2, 2))

plt.tight_layout()


# In[ ]:





# In[ ]:





#!/home/dennisyuan/.virtualenvs/deploy/bin/python3.5


from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np


#Lingting packages

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy.random as nprnd
import random
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import tree
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage import measure
from skimage import transform
import math
from scipy import ndimage
from skimage import feature
from skimage import filters
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy as scipy_entropy
from skimage import morphology
import pickle


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')

UPLOAD_FOLDER = '/home/dennisyuan/mysite/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'clf.pkl')
with open(pickle_file_path,'rb') as pickle_file1:
    clf = pickle.load(pickle_file1)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'svm.pkl')
with open(pickle_file_path,'rb') as pickle_file2:
    SVM = pickle.load(pickle_file2)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'lg.pkl')
with open(pickle_file_path,'rb') as pickle_file3:
    lg = pickle.load(pickle_file3)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'rf.pkl')
with open(pickle_file_path,'rb') as pickle_file4:
    rf = pickle.load(pickle_file4)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'knn.pkl')
with open(pickle_file_path,'rb') as pickle_file5:
    knn = pickle.load(pickle_file5)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'ada.pkl')
with open(pickle_file_path,'rb') as pickle_file6:
    ada = pickle.load(pickle_file6)

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'svmfinal.pkl')
with open(pickle_file_path,'rb') as pickle_file7:
    svmfinal = pickle.load(pickle_file7)


ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg','tiff','bmp'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/clf', methods=['GET', 'POST'])
def upload_fileclf():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            test = image_processing(file_loc)
            pre = clf.predict(test)
            if pre == 1:
                return "malanoma"
            else:
                return "not malanoma"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload Image File, Please put the mole on the center of the image not attaching the edge. </h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/svm', methods=['GET', 'POST'])
def upload_filesvm():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            test = image_processing(file_loc)
            pre = SVM.predict(test)
            if pre == 1:
                return "melanoma"
            else:
                return "not melanoma"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload Image File, Please put the mole on the center of the image not attaching the edge. </h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            test = image_processing(file_loc)
            pre1 = clf.predict_proba(test)
            pre2 = lg.predict_proba(test)
            pre3 = knn.predict_proba(test)
            pre4 = SVM.predict_proba(test)
            pre5 = rf.predict_proba(test)
            pre6 = ada.predict_proba(test)
            prob = np.zeros((1,6))
            prob[0,0] = pre1[0,1]
            prob[0,1] = pre2[0,1]
            prob[0,2] = pre3[0,1]
            prob[0,3] = pre4[0,1]
            prob[0,4] = pre5[0,1]
            prob[0,5] = pre6[0,1]
            pre = svmfinal.predict(prob)
            if pre == 1:
                return "melanoma"
            else:
                return "not melanoma"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload Image File, Please put the mole on the center of the image not attaching the edge.  </h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''




def image_processing(filename):
    Mean = [69236.576967,0.655035,1578.159566,0.386737,7060.628560,36.427073,36.427073,5390.919062,8245.273620,4206.274002,161.599501,81.116235,25.355420,-0.008700,15.804177,527.530438,157.804323,37.904037]
    Std = [41353.683102,0.145150,737.243555,0.148279,5111.862694,1569.445482,1569.445482,3321.720194,5322.786694,2645.578408,36.732585,23.307368,15.158453,0.779775,0.934235,238.389958,20.667027,11.669028]
    a = 0
    featur = np.zeros((1,18))

    img=mpimg.imread(filename)
    img= transform.resize(img, (576,767))
    gray = rgb2gray(img)
    #plt.imshow(gray,cmap = "gray")
    #plt.title('Gray Scale Image')
    # Plot the histogram of intensity
    Newgray = gray*255
    #plt.hist(Newgray.ravel(),256,[0,256])
    #plt.show()
    # Otsu thresholding
    bw1 = filters.threshold_otsu(gray)
    bw1 = gray <= bw1
    #plt.imshow(res,cmap = "gray")
    # Apply to open to disconnect from hairs and stuff
    bw1 = np.uint8(bw1)
    kernel = np.ones((5,5),np.uint8)
    bw2 = morphology.binary_opening(bw1,kernel)
    #plt.imshow(bw2,cmap = 'gray')
    # Get rid of small objects
    #bw2 = np.uint8(bw2)

    bw3 = morphology.remove_small_objects(bw2,1000)
    #find all your connected components (white blobs in your image)
    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw2, connectivity=8)

    #connectedComponentswithStats yields every seperated component with information on each of them, such as sizes
    #the following part is just taking out the background which is also considered a component, but most of that.
    #sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    #min_size = 150

    #your answer image
    #bw3 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    #for i in range(0, nb_components):
        #if sizes[i] >= min_size:
            #bw3[output == i + 1] = 255

    #plt.imshow(bw3,cmap= 'gray')
    #Fill the hole inside the mole
    bw4 = ndi.binary_fill_holes(bw3)
    bw4= np.uint8(bw4)
    #plt.imshow(bw4,cmap = 'gray')
    # Invert the image to get the mole
    bw5 = np.ones((bw4.shape),dtype = int) - bw4
    #plt.imshow(bw5,cmap = 'gray')

    # Fill the obejctive to get the mole
    bw6 = ndi.binary_fill_holes(bw5,structure=np.ones((8,8)))
    #plt.imshow(bw6,cmap= 'gray')
    bw6 = np.uint8(bw6)

    # Dot product to get the mole
    bw7 = np.multiply(bw6,bw4)
    #plt.imshow(bw7,cmap = 'gray')

    # To get rid of small object once again
    bw8 = np.uint8(bw7)
    #bw8 = morphology.remove_small_objects(bw7,2000)
    #find all your connected components (white blobs in your image)
    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw7, connectivity=8)

    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but n't want that.
    #sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    #min_size = 2000

    #your answer image
    #bw8 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    #for i in range(0, nb_components):
        #if sizes[i] >= min_size:
            #bw8[output == i + 1] = 1

    #plt.imshow(bw8,cmap= 'gray')

    bw8 = np.uint8(bw8)
    area = [r.area for r in measure.regionprops(bw8)]
    featur[a,0] = area[0]
    featur[a,0] = (featur[a,0] - Mean[0])/Std[0]

    eccentricity = [r.eccentricity for r in measure.regionprops(bw8)]
    featur[a,1] = eccentricity[0]
    featur[a,1] = (featur[a,1] - Mean[1])/Std[1]

    perimeter = [r.perimeter for r in measure.regionprops(bw8)]
    featur[a,2] = perimeter[0]
    featur[a,2] = (featur[a,2] - Mean[2])/Std[2]

    circularity = (4*math.pi*area[0])/(np.round(perimeter)**2)
    featur[a,3] =circularity[0]
    featur[a,3] = (featur[a,3] - Mean[3])/Std[3]

    inertia_tensor = [r.inertia_tensor for r in measure.regionprops(bw8)]
    pa = inertia_tensor[0]
    featur[a,4] = pa[0,0]
    featur[a,5] = pa[0,1]
    featur[a,6] = pa[1,0]
    featur[a,7] = pa[1,1]


    inertia_tensor_eigvals = [r.inertia_tensor_eigvals for r in measure.regionprops(bw8)]
    inertia_tensor_eigvals = inertia_tensor_eigvals[0]
    featur[a,8] = inertia_tensor_eigvals[0]
    featur[a,9] = inertia_tensor_eigvals[1]

    max_intensity = [r.max_intensity for r in measure.regionprops(bw8,Newgray)]
    featur[a,10] =max_intensity[0]

    mean_intensity = [r.mean_intensity for r in measure.regionprops(bw8,Newgray)]
    featur[a,11] =mean_intensity[0]

    min_intensity = [r.min_intensity for r in measure.regionprops(bw8,Newgray)]
    featur[a,12] =min_intensity[0]

    orientation = [r.orientation for r in measure.regionprops(bw8,Newgray)]
    featur[a,13] =orientation[0]

    entropy = scipy_entropy(bw8.ravel(),base = 2)
    featur[a,14] = entropy

    variance = ndimage.variance(Newgray,bw8)
    featur[a,15] = variance

    featur[a,16] = np.mean(Newgray)
    featur[a,17] = np.std(Newgray)

    featur[a,4] = (featur[a,4] - Mean[4])/Std[4]
    featur[a,5] = (featur[a,5] - Mean[5])/Std[5]
    featur[a,6] = (featur[a,6] - Mean[6])/Std[6]
    featur[a,7] = (featur[a,7] - Mean[7])/Std[7]
    featur[a,8] = (featur[a,8] - Mean[8])/Std[8]
    featur[a,9] = (featur[a,9] - Mean[9])/Std[9]
    featur[a,10] = (featur[a,10] - Mean[10])/Std[10]
    featur[a,11] = (featur[a,11] - Mean[11])/Std[11]
    featur[a,12] = (featur[a,12] - Mean[12])/Std[12]
    featur[a,13] = (featur[a,13] - Mean[13])/Std[13]
    featur[a,14] = (featur[a,14] - Mean[14])/Std[14]
    featur[a,15] = (featur[a,15] - Mean[15])/Std[15]
    featur[a,16] = (featur[a,16] - Mean[16])/Std[16]
    featur[a,17] = (featur[a,17] - Mean[17])/Std[17]
    return featur








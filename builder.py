from skimage.segmentation import slic,mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.measure import regionprops
from skimage.color import rgb2hsv
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold,ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import cv2 as cv 
import os,glob, shutil
import numpy as np
import argparse


def preprocessing(imagename, labeling,n_segments=100,sigma=5):
    # load the image and convert it to a floating point data type
    image= img_as_float(io.imread(imagename))
    #load the image with the labeling
    imlabels= cv.imread(labeling)
    imlabels= cv.resize(imlabels,(image.shape[1],image.shape[0]))
    # Perform slic super pixels segmentations
    segments = slic(image, n_segments = n_segments, sigma = sigma)
    regionpro=regionprops(segments,imlabels[:,:,1])
    prob=np.array([p.mean_intensity/255 for p in regionpro])
    prob = np.insert(prob, 0, [0])
    labels=[0 if i<0.5 else 1 for i in prob]
    return image,segments,labels

def make3d(descriptor):
    if (descriptor.ndim==2):
        dim=1
    else:
        dim=descriptor.shape[2]
    return np.reshape(descriptor, (descriptor.shape[0],descriptor.shape[1],dim))


def compute_polling(plane, segments):
    numsegments = int(np.max(segments)+1)
    fv = np.zeros((5,numsegments))
    fv[3,:] = np.max(plane)
    fv[4,:] = np.min(plane)
    for y in range(plane.shape[0]):
        for x in range(plane.shape[1]):
            L = int(segments[y,x])
            V = plane[y,x]
            fv[0,L] = fv[0,L]+1
            fv[1,L] = fv[1,L]+V
            fv[2,L] = fv[2,L]+V**2
            fv[3,L] = min(fv[3,L],V)
            fv[4,L] = max(fv[4,L],V)
    #for L in range(numsegments):
    fv[1,:] = fv[1,:]/fv[0,:]
    fv[2,:] = fv[2,:]-fv[1,:]**2
    fv[np.isnan(fv)] = 0
    return fv


def build_features(image, segments):
    hsv = rgb2hsv(image)
    h=hsv[:,:,0]/180
    s=hsv[:,:,1]/255
    v=hsv[:,:,2]/255
    ga=cv.GaussianBlur(image,(61,61),8.0)
    laga=cv.Laplacian(ga,cv.CV_64F)
    sobel= cv.Sobel(image,cv.CV_64F,1,0,ksize=3)
    descriptors=[image,h,s,v,ga,laga,sobel]  
    dataset=np.zeros((5,segments.max()+1))
    count=0
    for descriptor in descriptors:
        descriptor = make3d(descriptor)
        for channel in range(0,descriptor.shape[2]):
            plane = descriptor[:,:,channel]
            fv=compute_polling(plane,segments)
            dataset=np.r_[dataset,fv]
    return dataset

def build_dataset(imagespath,labelingpath,n_segments=100,sigma=5):
    imlist = glob.glob(os.path.join(imagespath,'*.jpg'))
    data  = np.ndarray((0,80))
    labeling = []
    for im in imlist:
        image,segments,labels = preprocessing(im,labelingpath+im[10:],n_segments,sigma)
        dataset = np.transpose(build_features(image,segments))
        data = np.r_[data,dataset]
        labeling += labels
    return data,labeling

def classify(dataset,labels):
    k = 5
    n_splits=5
    skf = StratifiedKFold(n_splits=n_splits)
    skf_ss = StratifiedShuffleSplit(n_splits=n_splits)
    C=1
    classifiers={"knn":KNeighborsClassifier(n_neighbors=k),"svml":svm.SVC(kernel='linear', C=C)
                 ,"svmr":svm.SVC(kernel='rbf', C=C),
                   "boost":GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                             max_depth=2, random_state=0)}
    scores={}
    for k in classifiers.keys():
        sc= cross_val_score(classifiers[k],dataset,labels,cv=skf)
        sc2= cross_val_score(classifiers[k],dataset,labels,cv=skf_ss)
        scores[k]=[sc.mean(),sc.std()]
        scores["shuffling "+ k]=[sc2.mean(),sc2.std()]
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--classifiers", required = False, help = "Classifiers all by default. Please ")
    ap.add_argument("-t", "--training", required = False, help = "Path to the training Image")
    ap.add_argument("-g", "--ground", required = False, help = "Path to the ground truth")
    args = vars(ap.parse_args())
    imagespath = "horse/rgb/"
    labelingpath = "horse/figure_ground/"
    data,labels = build_dataset(imagespath,labelingpath)
    np.savetxt("data.txt", data)
    np.savetxt("labels.txt",labels)
    results=classify(data,labels)
    np.savetxt("results.txt",results)


main()
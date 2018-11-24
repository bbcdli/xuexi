import cv2
import cv
import numpy as np
import time
import numpy
import imutils
from PIL import Image
import os
import settings
settings.set_global()


def calc_histogram(img1,img2,color,bins,h):
    for ch, col in enumerate(color):
        hist_item1 = cv2.calcHist([img1],[ch],None,[256],[0,255])
        hist_item2 = cv2.calcHist([img2],[ch],None,[256],[0,255])
        cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
        sc= cv2.compareHist(hist_item1, hist_item2, cv.CV_COMP_CORREL)
        #printsc
        hist=np.int32(np.around(hist_item1))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)
    return sc

#hy: calc similarity of each of images in group one to any of images in group two
#get potential strong samples
def calc_similarity(dirs1,path1,dirs2,path2,thresh,incl_neg_simi):
    tmp_ = []
    corr_ = []

    for item1 in dirs1:
        for item2 in dirs2:

            img1 = cv2.imread(path1 + item1)
            img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)
            img2 = cv2.imread(path2 + item2)
            img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

            bins = np.arange(256).reshape(256, 1)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            h = np.zeros((300, 256, 3))
            # print '\ncompare correlation cad vs cam:',item1,'vs',item2
            corr = calc_histogram(img1, img2, color, bins, h)
            corr_.append(corr)
            if incl_neg_simi == 1:
                if corr > float(thresh) or corr < -float(thresh):
                    # cmd = 'cp ' + path1+item1 + ' ./tmp/resized/' + name + '/'
                    # print cmd
                    # os.system(cmd)
                    tmp_.append(item1)
                #print 'similar images:', item1, 'vs', item2, ', confidence:', corr
            else:
                if corr > float(thresh):
                    # cmd = 'cp ' + path1+item1 + ' ./tmp/resized/' + name + '/'
                    # print cmd
                    # os.system(cmd)
                    tmp_.append(item1)
                #print 'similar images:', item1, 'vs', item2, ', confidence:', corr
    tmp_1 = list(set(tmp_))
    count = len(tmp_1)

    return count,tmp_1,min(corr_),max(corr_)

def calc_similarity_low(dirs1,path1,dirs2,path2,thresh,incl_neg_simi):
    tmp_ = []
    corr_ = []

    for item1 in dirs1:
        for item2 in dirs2:

            img1 = cv2.imread(path1 + item1)
            img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)
            img2 = cv2.imread(path2 + item2)
            img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

            bins = np.arange(256).reshape(256, 1)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            h = np.zeros((300, 256, 3))
            # print '\ncompare correlation cad vs cam:',item1,'vs',item2
            corr = calc_histogram(img1, img2, color, bins, h)
            corr_.append(corr)
            if incl_neg_simi == 1:
                if corr < float(thresh) or corr > -float(thresh):
                    # cmd = 'cp ' + path1+item1 + ' ./tmp/resized/' + name + '/'
                    # print cmd
                    # os.system(cmd)
                    tmp_.append(item1)
                #print 'similar images:', item1, 'vs', item2, ', confidence:', corr
            else:
                if corr < float(thresh):
                    # cmd = 'cp ' + path1+item1 + ' ./tmp/resized/' + name + '/'
                    # print cmd
                    # os.system(cmd)
                    tmp_.append(item1)
                #print 'similar images:', item1, 'vs', item2, ', confidence:', corr
    tmp_1 = list(set(tmp_))
    count = len(tmp_1)

    return count,tmp_1,min(corr_),max(corr_)

def split_samples(tmpA,tmpB):
    strong_disturb_to_other_ = []
    strong_pos_ = []
    for i in xrange(len(tmpA)):
        if tmpA[i] in tmpB:
            strong_disturb_to_other_.append(tmpA[i])

        if tmpA[i] not in tmpB:
            strong_pos_.append(tmpA[i])
    strong_disturb_to_other = list(set(strong_disturb_to_other_))
    strong_pos = list(set(strong_pos_))
    print '\ntotal number of strong disturb to others (remove)', len(strong_disturb_to_other)
    print 'total number of strong positive samples (keep)', len(strong_pos)
    return strong_pos, strong_disturb_to_other


def compare_images_cv():
    '''
    #img1 = cv2.imread('./tmp/distor_in/seed/hinten_wb1.jpg')
    img1 = cv2.imread('./tmp/distor_in/seed/hinten_ori1.jpg')
    img1 = cv2.cvtColor(img1,cv.CV_BGR2HSV)

    img2 = cv2.imread('./tmp/distor_in/test/hinten_t01up.jpg')
    img2 = cv2.cvtColor(img2,cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    h = np.zeros((300,256,3))
    #h=np.flipud(h) #hy: create graph containing histogram
    #cv2.imwrite('./tmp/resized/histogram_comp1.png',h)

    print 'numeric - correlation (0-1): hinten:[0.09813]', calc_histogram(img1,img2,color,bins,h)
    #hinten vs vorn: 0.326939
    #hinten vs links: 0.606455
    #hinten vs rechts: 0.567914
    #hinten vs oben: 0.3777
    #hinten vs unten: 0.20288


    ##############################################################
    ##############################################################
    img1 = cv2.imread('./tmp/distor_in/seed/vorn_wb1.jpg')
    img1 = cv2.cvtColor(img1,cv.CV_BGR2HSV)

    #img2 = cv2.imread('./tmp/distor_in/seed/hinten_ori1.jpg')
    img2 = cv2.imread('./tmp/distor_in/test/vorn_t01.jpg')
    img2 = cv2.cvtColor(img2,cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): vorn:[-0.05267]', calc_histogram(img1,img2,color,bins,h)

    ##############################################################
    ##############################################################
    #img1 = cv2.imread('./tmp/distor_in/seed/links_wb1.jpg')
    #img1 = cv2.imread('./Data/rechts/rechts_wei_wb1_rz215_2_r22_22.jpg')
    img1 = cv2.imread('./Data/bi/bi2_rz79_d0_0.jpg')
    img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)

    img2 = cv2.imread('./Test_Images/testpkg4/mouse/mouse_t1_rz82_d28_0.jpg')
    #img2 = cv2.imread('./tmp/distor_in/test/rechts_wei_t01.jpg')
    img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): links:[links_tou:0.015633] [links_wei:0.0215645]', calc_histogram(img1, img2, color, bins,h)
    #links_wei cad vs rechts_whole cam:    0.08082667
    #rechts_wei cad vs rechts_whole cam:  -0.0863344
    #rechts_whole cad vs rechts_whole cam:-0.0898635
    #rechts_tou cad vs rechts_tou cam:    -0.04960
    #rechts_wei cad vs rechts_wei cam:    -0.12043359   > whole > tou
    #
    #links_tou cad vs links_tou cam: 0.0156337
    #links_tou cad vs rechts_tou cam: -0.0442323
    #links_tou cad vs rechts_wei cam: -0.113299
    #links_tou cam vs rechts_tou cam: 0.710417

    # links_wei cad vs links_wei cam:     0.0215645    > tou
    # links_wei cad vs rechts_wei cam:   -0.1107354
    # links_wei cad vs rechts_tou cam:   -0.0373123
    # links_wei cam vs rechts_wei cam:   -0.6116884

    ##############################################################
    ##############################################################
    img1 = cv2.imread('./tmp/distor_in/seed/rechts_wb1.jpg')
    img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)

    # img2 = cv2.imread('./tmp/distor_in/seed/hinten_ori1.jpg')
    img2 = cv2.imread('./tmp/distor_in/test/rechts_t01.jpg')
    img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): rechts:[-0.0898635]', calc_histogram(img1, img2, color, bins,h)
    #rechts cad vs links_wei cam: 0.0167359
    #rechts cad vs links_wei cad: 975494
    ##############################################################
    ##############################################################

    img1 = cv2.imread('./tmp/distor_in/seed/oben_wb1.jpg')
    img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)

    # img2 = cv2.imread('./tmp/distor_in/seed/hinten_ori1.jpg')
    img2 = cv2.imread('./tmp/distor_in/test/oben_t01.jpg')
    img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): oben:[0.0243857]', calc_histogram(img1, img2, color, bins,h)

    ##############################################################
    ##############################################################

    img1 = cv2.imread('./tmp/distor_in/seed/unten_wb1.jpg')
    img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)

    img2 = cv2.imread('./tmp/distor_in/test/unten_t01.jpg')
    img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): unten:[-0.1038895]', calc_histogram(img1, img2, color, bins,h)

    ##############################################################
    ##############################################################

    ##############################################################
    ##############################################################
    img1 = cv2.imread('./Data/jiazi/jiazi3_rz79_d0_0.jpg')
    #img1 = cv2.imread('./tmp/distor_in/seed/rechts_wb1.jpg')
    img1 = cv2.cvtColor(img1, cv.CV_BGR2HSV)

    img2 = cv2.imread('./Data/yao/yao2_rz79_d0_4_flippedX.jpg')
    #img2 = cv2.imread('./tmp/input_patches/neg/bg_w1_200.jpg')
    img2 = cv2.cvtColor(img2, cv.CV_BGR2HSV)

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    h = np.zeros((300, 256, 3))
    print 'numeric - correlation (0-1): rechts vs neg:[-0.013516, -0.126276]', calc_histogram(img1, img2, color, bins, h)
    '''
    ##############################################################
    ##############################################################
    input_folder_name = 'vorn'
    path1 = './Data/' + input_folder_name +'/'          #hy: input
    path3 = './Test_Images/testpkg3_white_200x200/' + input_folder_name + '/'  #hy: target
    #path3 = './Test_Images/testpkg3_white_200x200/' + input_folder_name[5:] + '/'  #hy: target

    #disturbances
    path2_1 = './Test_Images/testpkg3_white_200x200/hinten/'
    path2_2 = './Test_Images/testpkg3_white_200x200/oben/'
    path2_3 = './Test_Images/testpkg3_white_200x200/links/'
    path2_4 = './Test_Images/testpkg3_white_200x200/rechts/'
    path2_5 = './Test_Images/testpkg3_white_200x200/unten/'

    dirs1 = os.listdir(path1)
    #dirs1_ = os.listdir(path1)
    #dirs1 = dirs1_[2001:len(dirs1_)-1]
    #dirs1 = dirs1_[0:100]

    dirs3 = os.listdir(path3)
    dirs2_1 = os.listdir(path2_1)
    dirs2_2 = os.listdir(path2_2)
    dirs2_3 = os.listdir(path2_3)
    dirs2_4 = os.listdir(path2_4)
    dirs2_5 = os.listdir(path2_5)
    thresh13 = 0.1

    count, cmp13, corr_min, corr_max = calc_similarity(dirs1, path1, dirs3, path3, thresh13, 1)
    print 'min', corr_min, 'max', corr_max, 'total similar images, 1 vs target:', count

    thresh13 = corr_max - 0.1
    thresh12 = thresh13

    count,cmp13,corr_min,corr_max = calc_similarity(dirs1,path1,dirs3,path3,thresh13,1)
    print 'min',corr_min, 'max',corr_max,'total similar images, 1 vs target:', count

    if count > 0:
        #'''
        count, cmp12_1, corr_min, corr_max = calc_similarity(dirs1, path1, dirs2_1, path2_1, thresh12,1)
        print 'min', corr_min, 'max', corr_max, 'total similar images, 1 vs 2_1:', count

        count, cmp12_2, corr_min, corr_max = calc_similarity(dirs1,path1, dirs2_2, path2_2, thresh12,1)
        print 'min', corr_min, 'max', corr_max, 'total similar images, 1 vs 2_2:', count

        count, cmp12_3, corr_min, corr_max = calc_similarity(dirs1,path1, dirs2_3, path2_3, thresh12,1)
        print 'min', corr_min, 'max', corr_max,'total similar images, 1 vs 2_3:', count

        count, cmp12_4, corr_min, corr_max = calc_similarity(dirs1,path1, dirs2_4,path2_4, thresh12,1)
        print 'min', corr_min, 'max', corr_max,'total similar images, 1 vs 2_4:', count

        count, cmp12_5, corr_min, corr_max = calc_similarity(dirs1, path1, dirs2_5, path2_5, thresh12,1)
        print 'min', corr_min, 'max',corr_max,'total similar images, 1 vs 2_5:', count
        tmpA = cmp13   #hy: potential good samples

        strong, disturb = split_samples(tmpA, cmp12_1)
        strong, disturb = split_samples(strong, cmp12_2)
        strong, disturb = split_samples(strong, cmp12_3)
        strong, disturb = split_samples(strong, cmp12_4)
        strong, disturb = split_samples(strong, cmp12_5)

        print 'final strong samples', len(strong)
        print strong
        dest_path = './Data/strong/'
        for item1 in strong:
            #print item1
            cmd = 'cp ' + path1 + item1 + ' ' + dest_path
            os.system(cmd)
        print 'total images found:', len(os.listdir(dest_path))

        ################################ get weak neg #####
        strong, disturb = split_samples(tmpA, cmp12_1)
        strong, disturb = split_samples(disturb, cmp12_2)
        strong, disturb = split_samples(disturb, cmp12_3)
        strong, disturb = split_samples(disturb, cmp12_4)
        strong, disturb = split_samples(disturb, cmp12_5)

        print 'final strong disturb to other classes:', len(disturb)
        print disturb
        dest_path = './Data/disturb/'
        for item_weak in disturb:
            cmd = 'cp ' + path1 + item_weak + ' ' + dest_path
            os.system(cmd)
        print 'total images found:', len(os.listdir(dest_path))
        #'''
    else:
        print 'no samples filtered'

#compare_images_cv()


def get_weak_positive():
    ##############################################################
    input_folder_name = 'weak_pos_tmp'
    path1 = './Data/' + input_folder_name +'/'          #hy: input
    path3 = './Test_Images/testpkg3_white_200x200/' + 'hinten' + '/'  #hy: target

    dirs1 = os.listdir(path1)
    #dirs1_ = os.listdir(path1)
    #dirs1 = dirs1_[2001:len(dirs1_)-1]
    #dirs1 = dirs1_[0:100]

    dirs3 = os.listdir(path3)
    thresh13 = 0

    count, cmp13, corr_min, corr_max = calc_similarity(dirs1, path1, dirs3, path3, thresh13, 1)
    print 'min', corr_min, 'max', corr_max, 'total similar images, 1 vs target:', count

    thresh13 = corr_min + 0.01
    thresh12 = thresh13

    count, cmp13, corr_min, corr_max = calc_similarity_low(dirs1, path1, dirs3, path3, thresh13, 0)
    print 'min', corr_min, 'max', corr_max, 'total weak similar images, 1 vs target:less than',thresh13, count

    if count > 0:
        tmpA = cmp13   #hy: potential good samples
        dest_path = './Data/weak_pos/'
        for item1 in tmpA:
            #print item1
            cmd = 'mv ' + path1 + item1 + ' ' + dest_path
            os.system(cmd)
        print 'total weak positive images (remove):', len(os.listdir(dest_path))

    else:
        print 'no samples filtered'

get_weak_positive()

#import numpy
#from PIL import Image
#import cv2

def similarness(image1,image2):
    """
Return the correlation distance be1tween the histograms. This is 'normalized' so that
1 is a perfect match while -1 is a complete mismatch and 0 is no match.
"""
    # Open and resize images to 200x200
    i1 = Image.open(image1).resize((200,200))
    i2 = Image.open(image2).resize((200,200))

    # Get histogram and seperate into RGB channels
    i1hist = numpy.array(i1.histogram()).astype('float32')
    i1r, i1b, i1g = i1hist[0:256], i1hist[256:256*2], i1hist[256*2:]
    # Re bin the histogram from 256 bins to 48 for each channel
    i1rh = numpy.array([sum(i1r[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i1bh = numpy.array([sum(i1b[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i1gh = numpy.array([sum(i1g[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    # Combine all the channels back into one array
    i1histbin = numpy.ravel([i1rh, i1bh, i1gh]).astype('float32')

    # Same steps for the second image
    i2hist = numpy.array(i2.histogram()).astype('float32')
    i2r, i2b, i2g = i2hist[0:256], i2hist[256:256*2], i2hist[256*2:]
    i2rh = numpy.array([sum(i2r[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2bh = numpy.array([sum(i2b[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2gh = numpy.array([sum(i2g[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2histbin = numpy.ravel([i2rh, i2bh, i2gh]).astype('float32')

    return cv2.compareHist(i1histbin, i2histbin, 0)
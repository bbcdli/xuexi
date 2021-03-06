#hy create the global variable for width and height that will be resized (in read_images.py)
#hy:collection of all global variables
# hy create the global variable for width and height that will be resized (in read_images.py)
#import Image

#import cv2
import numpy as np

def set_global():
    global test_label_file_a
    global w_resize
    global h_resize
    global crop_x1
    global crop_x2
    global crop_y1
    global crop_y2
    global LABELS
    global LABEL_names
    global LABEL_short
    global Misclassified
    global CorrectClassified
    global maxNumSaveFiles
    global patches
    global tmp
    global tmp2
    global data
    global data_label_file
    global data_label_path
    global test_images
    global test_label_file
    global test_label_path
    global absolute_path


    h_resize = 184#64#184 #124 has seq error # 28 initialize a value, sometimes it will be changed a bit after performing "imutils.resize()"
    w_resize = 184#64#184  # it is still tricky to find proper initial values

    crop_x1 = 100  # [350:750,610:1300] #hy: [y1:y2,x1:x2]
    crop_x2 = 420
    crop_y1 = 100
    # crop_y2 = (crop_y1 + (crop_x2-crop_x1))*h_resize/w_resize #according to accepted shape by tensorflow
    crop_y2 = crop_y1 + (crop_x2 - crop_x1) * h_resize / w_resize  # according to accepted shape by tensorflow

    # Here notice to add '/' at the end of subdirectory name
    num_of_classes = 2

    if num_of_classes == 6:
        LABELS = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
        LABEL_names = ['hinten', 'links', 'oben', 'rechts','unten', 'vorn']
        LABEL_short = ['H', 'L', 'O', 'R','U', 'V']
    if num_of_classes == 82:
        LABELS = ['hinten_1/','hinten_2/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn_1/','vorn_2/']
        LABEL_names = ['hinten_1','hinten_2','links', 'oben', 'rechts', 'unten', 'vorn_1','vorn_2']
        LABEL_short = ['H1','H2','L', 'O', 'R', 'U', 'V1','V2']
    if num_of_classes == 7:
        LABELS = ['neg/','hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
        LABEL_names = ['neg','hinten', 'links', 'oben', 'rechts', 'unten', 'vorn']
        LABEL_short = ['N','H', 'L', 'O', 'R', 'U', 'V']
    if num_of_classes == 2:
        LABELS = ['good/', 'damaged/']
        LABEL_names = ['good', 'damaged']
        LABEL_short = ['g','d']
    if num_of_classes == 5:
        LABELS = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/']
        LABEL_names = ['hinten', 'links', 'oben', 'rechts','unten']
        LABEL_short = ['H','L', 'O', 'R','U']


    Misclassified = '../classified/Misclassified'
    CorrectClassified = '../classified/CorrectClassified'
    maxNumSaveFiles = 1000

    absolute_path = '/home/hy/Documents/fenlei_tf/script_Jan26/data'
    patches = absolute_path + './tmp/input_patches/'
    tmp = absolute_path + './tmp/'#
    tmp2 = absolute_path + './tmp/tmp2/'
    data = absolute_path + '/train'
    data_label_path = data + '/*/*'
    data_label_file = absolute_path + '/FileList.txt'


    #test_images = absolute_path + './Test_Images/testpkg2_no_bg/'
    #test_images = absolute_path + './Test_Images/testpkg3_white_200x200/'
    #test_images = absolute_path + './Test_Images/testpkg5_42x42/'
    #test_images = absolute_path + './Test_Images/testpkg5local_224x224/'
    #test_images = absolute_path + './Test_Images/testpkg7_mix_crop/'
    #test_images = absolute_path + './Test_Images/7_2/'  #mix sizes and dark
    #test_images = absolute_path + './Test_Images/testpkg8_frame/'
    #test_images = absolute_path + './Test_Images/testpkg8_dark/'
    test_images = absolute_path + './test/'


    test_label_path = test_images + '/*/*'
    test_label_file = absolute_path + '/FileList_TEST.txt'
    test_label_file_a = absolute_path + '/FileList_TEST_act1.txt'
    #test_images = absolute_path + './Test_Images/testpkg6big/'



'''
from PIL import Image #hy: create video with images
activation_test_img = Image.open('../hintenTest.jpg')
activation_test_img.show()
activation_test_img.save('../hintenTest2.jpg')


img = Image.open('../tmp/resized/rechts/rechts_t2_1_rz400_d0_0400_1.jpg')
bigsize = (img.size[0]*3, img.size[1]*3)
mask = Image.new('L', bigsize, 0)

draw = ImageDraw.Draw(mask)

draw.ellipse((0,0) + bigsize, fill=30)

mask = mask.resize(img.size, Image.ANTIALIAS)
#bg = ImageOps.fit(bgOri,mask.size, centering=(0.5,0.5))
img.putalpha(mask)
#print 'bg size', bg.shape()
img.save('../1_bg.jpg')


bg_in = cv2.imread('../tmp/resized/rechts/rechts_t2_1_rz400_d0_0400_1.jpg')
for alpha in np.arange(0,1.1, 0.1)[::-1]:
    back = Image.new('RBGA', bg_in.size)
    back.paste(bg_in)
    poly = Image.new('RGBA', (400,400))
    pdraw = ImageDraw.Draw(poly)

    back.paste(poly, (0,0), mask=poly)

    back.paste(back
    #bg = Image.fromarray(bg_out)
'''

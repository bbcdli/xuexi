########################################################################################################
#hy create the global variable for width and height that will be resized (in read_images.py)
#hy:collection of all global variables
# hy create the global variable for width and height that will be resized (in read_images.py)
from PIL import Image
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
    global test_label_path
    global test_images
    global test_label_file
    global test_label_path1
    global test_label_path2
    global test_label_path3
    global test_label_path4
    global test_label_path5
    global test_label_path6
    global test_label_path7
    global test_label_path8
    global test_label_path9
    global test_label_path10
    global absolute_path
    global LABELS_en

########################################################################################################

    h_resize = 184#64#184 #124 has seq error # 28 initialize a value, sometimes it will be changed a bit after performing "imutils.resize()"
    w_resize = 184#64#184  # it is still tricky to find proper initial values

    crop_x1 = 100  # [350:750,610:1300] #hy: [y1:y2,x1:x2]
    crop_x2 = 420
    crop_y1 = 100
    # crop_y2 = (crop_y1 + (crop_x2-crop_x1))*h_resize/w_resize #according to accepted shape by tensorflow
    crop_y2 = crop_y1 + (crop_x2 - crop_x1) * h_resize / w_resize  # according to accepted shape by tensorflow

    # Here notice to add '/' at the end of subdirectory name
    num_of_classes = 6

    if num_of_classes == 6:
        LABELS = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/', 'vorn/']
        LABELS_en = ['back/', 'left/', 'top/', 'right/', 'bottom/', 'front/']
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
        LABELS = ['vorn/', 'hinten/']
        LABEL_names = ['vorn', 'hinten']
        LABEL_short = ['L','U']
    if num_of_classes == 5:
        LABELS = ['hinten/', 'links/', 'oben/', 'rechts/', 'unten/']
        LABEL_names = ['hinten', 'links', 'oben', 'rechts','unten']
        LABEL_short = ['H','L', 'O', 'R','U']


    Misclassified = '../classified/Misclassified'
    CorrectClassified = '../classified/CorrectClassified'
    maxNumSaveFiles = 1000

    absolute_path = '../'
    patches = absolute_path + './tmp/input_patches/'
    tmp = absolute_path + './tmp/'#
    tmp2 = absolute_path + './tmp/tmp2/'
    data = absolute_path + '/Data/top_data/training/'   # './Test_Images/MA_test_6cl/'#top_data/training/' #
    data_label_path = data + '/*/*'
    data_label_file = absolute_path + './FileList.txt'

    test_label_path1 = absolute_path + './Test_Images/testpkg2_no_bg/' + '/*/*'
    test_label_path2 = absolute_path + './Test_Images/testpkg3_white_200x200/' + '/*/*'
    test_label_path3 = absolute_path + './Test_Images/testpkg5_42x42/' + '/*/*'
    test_label_path4 = absolute_path + './Test_Images/testpkg5local_224x224/' + '/*/*'
    test_label_path5 = absolute_path + './Test_Images/testpkg7_2/' + '/*/*'
    test_label_path6 = absolute_path + './Test_Images/testpkg7small/' + '/*/*' #mix sizes and dark
    test_label_path7 = absolute_path + './Test_Images/testpkg9_frame/' + '/*/*' #testpkg9_frame
    test_label_path8 = absolute_path + './Test_Images/testpkg8_dark/' + '/*/*'
    test_label_path9 = absolute_path + './Test_Images/MA_test_6cl/' + '/*/*'
    test_label_path10 = absolute_path + './Test_Images/MA_testpkg_tfseg_1/' + '/*/*'


    test_label_path = test_label_path2
    test_label_file = absolute_path + './FileList_TEST.txt'
    test_label_file_a = absolute_path + './FileList_TEST_act1.txt'
    #test_images = absolute_path + './Test_Images/testpkg6big/'

#http://rnd.azoft.com/object-detection-fully-convolutional-neural-networks/

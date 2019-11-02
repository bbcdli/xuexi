import glob
import os,sys
import settings
# activate global var
settings.set_global()
DEBUG = True


# Data
LABEL_LIST = settings.data_label_file
LABEL_PATH = settings.data_label_path

LABEL_LIST_TEST = settings.test_label_file
LABEL_PATH_TEST = settings.test_label_path

LABELS = settings.LABELS  # hy
LABEL_names = settings.LABEL_names  # hy

def prepare_list(image_list_file, image_label_path):
    filelist = sorted(glob.glob(image_label_path))  # hy: sorted is used for doing active fields analysis
    Output_file = image_list_file

    # print 'image_label_path',image_label_path

    if DEBUG:
        print(filelist)

    file_name_label = []

    # method 1
    # label_list = os.listdir(class_PATH)
    # print "class list:", label_list

    # method 2 same result as 1
    # class_label = next(os.walk(class_PATH))[1]
    # print "class_label:", class_label
    if DEBUG:
        if os.path.isfile(Output_file) == False:
            print("file not found, please create one empty file first")
            open(Output_file)
        else:
            print("file found OK")

    for filename in filelist:
        class_index = 0
        for label in LABELS:  # hy search files under this path
            # label = class_PATH + label  #hy ../Data/2
            if str.find(filename, label) != -1:  # hy find all lines containing /Data/class_index
                file_name_label.append(filename + " " + str(class_index))
            # print file_name_label
            # else:
            #    print 'no folder found'
            class_index = class_index + 1

    lines = "\n".join(file_name_label)

    # write lines into the file
    with open(Output_file, 'w') as f:
        f.writelines(lines)

    # print "first line:"
    # with open(Output_file, 'r') as f:
    #    plines = [next(f) for x in range(1)]
    #    print plines

    if DEBUG:
        # method 1
        print("method 1: file length:", sum(1 for line in open(Output_file)))

        # method 2
        with open(Output_file) as f:  # use "with sth as" to define a file
            print("method 2: file length:", sum(1 for line in f))

        # method 3
        with open(Output_file) as f:
            file_length = len(f.readlines())
            print("method 3: file length:", file_length)

    print('file list is created.', image_list_file, 'path', image_label_path)


image_list_file, image_label_path = settings.data_label_file, settings.data_label_path

prepare_list(image_list_file,image_label_path)
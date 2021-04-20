import numpy as np
import os
import glob
import cv2
import argparse
import time
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

import pickle

# Add openpose to system PATH and import
#sys.path.append('/home/tremlak/openpose/build/python');
#from openpose import pyopenpose as op

# This program turns labels.txt into labels.npy

def main():
    path = "/Users/jaredwilliams/Documents/AI/GolfCart/"
    #structure_form(path)
    #op_compile(path)
    #labels_text_to_array(path)
    compile_data(path)

def structure_form(project_path):

   
    # for video i in videos_path:
        # creates a folder set-i
        # creates int-i.txt in set-i
        # creates images-i in set-i
        # splits the video i into images and puts them in images-i
        # run all the images in images-i through openpose and create op-n.npy in set-i
    videos = project_path + "input-videos/*.avi"
    #print(videos)
    file_count = 0
    for video_file in glob.glob(videos):
        print("new video")
        #print(os.path.dirname(video_file))
        set_path = project_path + "Set" + str(file_count)
        os.mkdir(set_path)
        images_path = set_path + "/images-" + str(file_count)
        os.mkdir(images_path)
        

        #split videofile
        # Path to video file
        vidObj = cv2.VideoCapture(video_file)

        # Used as counter variable
        count = 0

        # checks whether frames were extracted
        success = 1

        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()

            # Saves the frames with frame-count
            cv2.imwrite(images_path + "/frame%d.jpg" % count, image)

            count += 1
        os.chdir(set_path)
        label_text = open("int-" + str(file_count) + ".txt", "w")
        file_count += 1

        ## runs images_path folder through openpose:

def op_compile(project_path):

    videos = project_path + "input-videos/*.avi"
    #print(videos)
    file_count = 0
    for video_file in glob.glob(videos):
        print("new video")
        # print(os.path.dirname(video_file))
        set_path = project_path + "Set" + str(file_count)
        images_path = set_path + "/images-" + str(file_count)
        op_images_path = set_path + "/op_images-" + str(file_count)
        #os.mkdir(op_images_path)

        #################
        # Setup for OpenPose processing.
        #########
        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", default="{}/".format(images_path), help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "/home/tremlak/openpose/models/"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        ###################################################
        ##############################
        # Starting OpenPose
        #########
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Read frames on directory
        imagePaths = op.get_images_on_directory(args[0].image_dir);
        start = time.time()

        count = 0
        # Process and display images
        np_list = []
        for imagePath in imagePaths:
            #print(imagePath)
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            if (imageToProcess is not None):
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                if (len(datum.poseKeypoints.shape) == 0):
                    array_addition = np.zeros(75)
                else:
                    # TODO: Fix for multiple people in frame.
                    array_addition = datum.poseKeypoints[0]
                    array_addition = np.reshape(array_addition, 75)

                np_list.append(array_addition)

                #print("Body keypoints: \n" + str(datum.poseKeypoints))

                if not args[0].no_display:
                    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
                    key = cv2.waitKey(15)
                    if key == 27: break
                # Save OpenPose output images.
                cv2.imwrite('{}/frame{}.jpg'.format(op_images_path, count),datum.cvOutputData)
            else:
                print(count)
            count += 1

        end = time.time()
        # Save npy array
        np_array = np.array(np_list)

        np.save('{}'.format(set_path + "/op-{}.npy".format(str(file_count))), np_array)
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

        ########################################################

        file_count += 1


def labels_text_to_array(project_path):
    sets = project_path + "Set*"

    for set_path in glob.glob(sets):
        tag = os.path.basename(set_path)[3:]
        f = open(set_path + "/int-" + tag + ".txt", "r")

        contents = f.read()

        contents = contents.split("\n")
        image_count = contents[0]
        intervals = contents[1:]

        for i in range(len(intervals)):
            intervals[i] = intervals[i].split("-")

        fill_arr = np.zeros(int(image_count))
        print(intervals)
        for interval in intervals:
            if (len(interval) > 1):
                for subsection in range(int(interval[0]), int(interval[1]) + 1):
                    fill_arr[subsection] = 1

        np.save(set_path + "/labels-" + tag, fill_arr)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def compile_data(project_path):
    # create list "data_build"
    # create list "labels_build"
    # for sets in data bank:
        # import labels-i.npy as numpy array unsplit_labels
        # import op_data-1.npy as numpy array unsplit_op
        # add 1/10 of unsplit_labels to data_build
        # add 1/10 of unsplit_op to labels_build
    # split data_build
    # save train_data and test_data

    # split labels_build
    # save train_data and test_data
    sets = project_path + "Set*"
    data_build = None
    labels_build = None

    for set_path in glob.glob(sets):
        tag = os.path.basename(set_path)[3:]

        labels_load = np.load(set_path + "/labels-" + tag + ".npy")
        labels_load = labels_load.flatten()
        data_load = np.load(set_path + "/op-" + tag + ".npy")
        data_load = np.reshape(data_load, (-1, 75))
        if labels_build is not None:
            labels_build = np.concatenate((labels_build, labels_load))
            data_build = np.concatenate((data_build, data_load))
        else:
            labels_build = labels_load
            data_build = data_load

    labels_build = labels_build[0::4]
    data_build = data_build[0::4]

    #X_train, X_test, y_train, y_test = train_test_split(data_build, labels_build, test_size = 0.10, random_state = 42)
    split_index = int(data_build.shape[0] * .9)

    X_train = data_build[:split_index]
    X_test = data_build[split_index:]

    y_train = labels_build[:split_index]
    y_test = labels_build[split_index:]

    print("data compiled together")

    filename = 'finalized_model.sav'

    space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
             'max_depth': hp.quniform('max_depth', 10, 1200, 10),
             'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
             'min_samples_leaf': hp.uniform('min_samples_leaf', 0, .5),
             'min_samples_split': hp.uniform('min_samples_split', 0, 1),
             'n_estimators': hp.choice('n_estimators', [50, 300, 500, 750, 1200])
             }

    def objective(space):
        model = RandomForestClassifier(criterion=space['criterion'],
                                       max_depth=space['max_depth'],
                                       max_features=space['max_features'],
                                       min_samples_leaf=space['min_samples_leaf'],
                                       min_samples_split=space['min_samples_split'],
                                       n_estimators=space['n_estimators'],
                                       )

        accuracy = cross_val_score(model, X_train, y_train, cv=4).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)

    print("best type:", type(best))
    print(best)

    crit = {0: 'entropy', 1: 'gini'}
    feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
    est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200}

    trainedforest = RandomForestClassifier(criterion=crit[best['criterion']],
                                           max_depth=best['max_depth'],
                                           max_features=feat[best['max_features']],
                                           min_samples_leaf=best['min_samples_leaf'],
                                           min_samples_split=best['min_samples_split'],
                                           n_estimators=est[best['n_estimators']]
                                           ).fit(X_train, y_train)
    predictionforest = trainedforest.predict(X_test)
    print(confusion_matrix(y_test, predictionforest))
    print(classification_report(y_test, predictionforest))
    acc5 = accuracy_score(y_test, predictionforest)
    print(acc5)

    """
    ada_boost = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    grad_boost = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    xgb = xgboost.XGBRegressor(random_state=42)
    """

    """
    ada_boost.fit(X_train, y_train)
    grad_boost.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    """

    """
    ada_boost_score = ada_boost.score(X_test, y_test)
    grad_boost_score = grad_boost.score(X_test, y_test)
    xgb_boost_score = xgb.score(X_test, y_test)
    """

    #print()
    #print("Random Forest Score:", rnd_fst_score)
    """
    print("AdaBoost Score:", ada_boost_score)
    print("Gradient Boost Score:", grad_boost_score)
    print("XGBoost Score:", xgb_boost_score)
    """
    filename = 'full_dataset_model.sav'
    pickle.dump(trainedforest, open(filename, 'wb'))



if __name__ == '__main__':
    main()

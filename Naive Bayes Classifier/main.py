### TEST FUNCTION: test_question1
# DO NOT REMOVE THE ABOVE LINE
import numpy as np
import scipy.io
import math
import geneNewData

global train0, train1, test0, test1

def main():
    global train0, train1, test0, test1
    myID = '5161'  # change to last 4 digit of your studentID
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train' + myID + '.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train' + myID + '.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset' + '.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset' + '.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')

    # Task 1
    avg_brightness_train0, std_brightness_train0 = extract_features(train0)
    avg_brightness_train1, std_brightness_train1 = extract_features(train1)

    # print(train0.shape[0])
    # print(np.mean(avg_brightness_train0))
    # Task 2
    mean_var_params_digit0 = calculate_parameters(np.vstack((avg_brightness_train0, std_brightness_train0)).T)
    mean_var_params_digit1 = calculate_parameters(np.vstack((avg_brightness_train1, std_brightness_train1)).T)
    # print(mean_var_params_digit0)

    # Task 3
    # Implement naive_bayes_classifier function and use it to predict labels for test0 and test1

    avg_brightness_test0, std_brightness_test0 = extract_features(test0)
    test0 = np.vstack((avg_brightness_test0, std_brightness_test0)).T

    avg_brightness_test1, std_brightness_test1 = extract_features(test1)
    test1 = np.vstack((avg_brightness_test1, std_brightness_test1)).T

    predicted_labels_digit0 = naive_bayes_classifier(test0, mean_var_params_digit0, mean_var_params_digit1)
    predicted_labels_digit1 = naive_bayes_classifier(test1, mean_var_params_digit0, mean_var_params_digit1)

    # print(predicted_labels_digit0)
    accuracy_digit0 = calculate_accuracy(np.zeros(len(test0)), predicted_labels_digit0)
    accuracy_digit1 = calculate_accuracy(np.ones(len(test1)), predicted_labels_digit1)

    # print(accuracy_digit1)
    # # # Output
    output_list = ['5161'] + list(mean_var_params_digit0) + list(mean_var_params_digit1) + [accuracy_digit0, accuracy_digit1]
    print(output_list)



def extract_features(images):
    avg_brightness = np.mean(images, axis=(1, 2))  # Calculate average brightness
    std_brightness = np.std(images, axis=(1, 2))  # Calculate standard deviation of brightness
    return avg_brightness, std_brightness


def calculate_parameters(images):
    mean_feature1 = np.mean(images[:, 0])
    var_feature1 = np.var(images[:, 0])

    mean_feature2 = np.mean(images[:, 1])
    var_feature2 = np.var(images[:, 1])

    return mean_feature1, var_feature1, mean_feature2, var_feature2


def naive_bayes_classifier(test_data, params0, params1):
    global train0, train1, test0, test1
    mean_feature1_digit0, var_feature1_digit0, mean_feature2_digit0, var_feature2_digit0 = params0[:4]
    mean_feature1_digit1, var_feature1_digit1, mean_feature2_digit1, var_feature2_digit1 = params1[:4]

    # Calculate likelihoods for digit0
    likelihood_digit0 = (
                                np.exp(-((test_data[:, 0] - mean_feature1_digit0) ** 2) / (
                                            2 * var_feature1_digit0)) / np.sqrt(2 * np.pi * var_feature1_digit0)
                        ) * (
                                np.exp(-((test_data[:, 1] - mean_feature2_digit0) ** 2) / (
                                            2 * var_feature2_digit0)) / np.sqrt(2 * np.pi * var_feature2_digit0)
                        )

    # print(likelihood_digit0)

    # Calculate likelihoods for digit1
    likelihood_digit1 = (
                                np.exp(-((test_data[:, 0] - mean_feature1_digit1) ** 2) / (
                                            2 * var_feature1_digit1)) / np.sqrt(2 * np.pi * var_feature1_digit1)
                        ) * (
                                np.exp(-((test_data[:, 1] - mean_feature2_digit1) ** 2) / (
                                            2 * var_feature2_digit1)) / np.sqrt(2 * np.pi * var_feature2_digit1)
                        )

    # print(likelihood_digit1)
    # # Calculate priors
    prior_digit0 = len(train0) / (len(train0) + len(train1))
    prior_digit1 = len(train1) / (len(train0) + len(train1))
    #
    # # Calculate posteriors
    posterior_digit0 = likelihood_digit0 * prior_digit0
    posterior_digit1 = likelihood_digit1 * prior_digit1
    #
    # # Predict labels
    predicted_labels = np.argmax(np.vstack((posterior_digit0, posterior_digit1)), axis=0)

    return predicted_labels

def calculate_accuracy(true_labels, predicted_labels):
    return np.sum(true_labels == predicted_labels) / len(true_labels)

if __name__ == '__main__':
    main()
import os
import json
import glob
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--data_path", type=str, help='path to dataset')
args = parser.parse_args()

def msu_process(data_dir, label_save_dir):
    test_list = []
    for line in open(data_dir + 'MSU-MFSD/test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open(data_dir + 'MSU-MFSD/train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    train_final_json = []
    test_final_json = []
    all_final_json = []
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    dataset_path = data_dir + 'MSU-MFSD/'
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        video_num = path_list[i].split('/')[-2].split('_')[1][-2:]
        if (video_num in train_list):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()



def casia_process(data_dir, label_save_dir):
    train_final_json = []
    test_final_json = []
    all_final_json = []
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    dataset_path = data_dir + 'CASIA_faceAntisp/'
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if path_list[i].find('/train_release/')!=-1:
            train_final_json.append(dict)
        elif path_list[i].find('/test_release/')!=-1:
            test_final_json.append(dict)
        all_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()



def idiap_process(data_dir, label_save_dir):
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_print = open(label_save_dir + 'print_label.json', 'w')
    f_video = open(label_save_dir + 'video_label.json', 'w')
    dataset_path = data_dir + 'Idiap-Replayattack/'
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        elif path_list[i].find('/enroll/') != -1:
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].find('/train/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/devel/') != -1):
            valid_final_json.append(dict)
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)

        all_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()

def oulu_process(data_dir, label_save_dir):
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    dataset_path = data_dir + 'Oulu-NPU/'
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = int(path_list[i].split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        all_final_json.append(dict)
        if (path_list[i].find('/Train_files/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/Dev_files/') != -1):
            valid_final_json.append(dict)
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)

    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()






if __name__=="__main__":
    msu_process(args.data_path + '/', './labels/msu/')
    casia_process(args.data_path + '/', './labels/casia/')
    idiap_process(args.data_path + '/', './labels/idiap/')
    oulu_process(args.data_path + '/', './labels/oulu/')
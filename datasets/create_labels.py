import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='./second_data/', help='path to dataset root dir')
parser.add_argument('--train', default='train', help='path to training dataset')
parser.add_argument('--test', default='test', help='path to test dataset')
parser.add_argument('--train_labels', default='train.txt', help='path to training dataset')
parser.add_argument('--test_labels', default='test.txt', help='path to test dataset')
opt = parser.parse_args()

train_folder = os.path.join(opt.data_root, opt.train)
test_folder = os.path.join(opt.data_root, opt.test)
train_file = os.path.join(opt.data_root, opt.train_labels)
test_file = os.path.join(opt.data_root, opt.test_labels)

def writingLabels(status = 'train'):
    if status=='train':
        _folder = train_folder
        _file = train_file
    else:
        _folder = test_folder
        _file = test_file
    with open(_file, 'w+') as f_loader:
        image_list = os.listdir(_folder)
        image_list.sort()
        for image_name in image_list:
            label = image_name.split('_')[0]
            writtenLine = image_name + ' ' + label + '\n'
            f_loader.write(writtenLine)

def main():
    writingLabels('train')
    writingLabels('test')

if __name__ == "__main__":
    main()
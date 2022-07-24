import os
import json
import itertools

def create_cifar_dataset_description(dataset_path='./data/cifar-10', train_description_path='./data/cifar-10/train.json', test_description_path='./data/cifar-10/test.json'):

    def create_image_description(dataset='cifar-10', subdataset=None, image_path=None, size=None, class_name=None):
        return {
            'dataset': dataset,
            'subdataset': subdataset,
            'image_path': image_path,
            'size': size,
            'class_name': class_name,
            'image_name': '_'.join([dataset, subdataset, class_name, image_path.split('/')[-1]])
        }

    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    train = list()
    test = list()
    size=(32, 32, 3)
    for c in classes:
        train_path = '/'.join([dataset_path, 'train', c])
        test_path = '/'.join([dataset_path, 'test', c])
        print(train_path)
        print(test_path)
        train_images = os.listdir(train_path)
        test_images = os.listdir(test_path)
        tmp_train = [0] * len(train_images)
        tmp_test = [0] * len(test_images)
        for i, image in enumerate(train_images):
            image_path = '/'.join([train_path, image])
            tmp_train[i] = create_image_description(subdataset='train', image_path=image_path, size=size, class_name=c)
        for i, image in enumerate(test_images):
            image_path = '/'.join([test_path, image])
            tmp_test[i] = create_image_description(subdataset='test', image_path=image_path, size=size, class_name=c)
        train.append(tmp_train)
        test.append(tmp_test)
    train = list(itertools.chain(*train))
    test = list(itertools.chain(*test))
    with open(train_description_path, 'w') as f:
        json.dump(train, f, indent=4)
    with open(test_description_path, 'w') as f:
        json.dump(test, f, indent=4)


if __name__ == '__main__':
    create_cifar_dataset_description()
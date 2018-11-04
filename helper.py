import re
import random
import numpy as np
import cv2
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def transform_scale(img, ratio = None, ratio_set = [0.9, 1.1]):
    if ratio is None:
        ratio = random.sample(ratio_set, 1)[0]
    
    img_c = np.copy(img)
    rows,cols,ch = img_c.shape
    
    img1 = cv2.resize(img_c,(int(ratio*cols), int(ratio*rows)), interpolation = cv2.INTER_CUBIC)
    rows1,cols1,ch1 = img1.shape
    
    result = np.zeros_like(img_c)
    if (ratio >= 1.0):
        result = img1[(rows1 - rows)//2:(rows1 - rows)//2+rows, (cols1 - cols)//2:(cols1 - cols)//2+cols, :]
    else:
        result[(rows - rows1)//2:(rows - rows1)//2+rows1, (cols - cols1)//2:(cols - cols1)//2+cols1, :] = img1
    
    return result


def transform_rotate(img, angle = None, lower = -15,upper = 15, scale=1.0):
    if angle is None:
        angle = random.randint(lower, upper)
    
    img_c = np.copy(img)
    rows,cols,ch = img_c.shape
    center = (cols//2,rows//2) 
    
    M = cv2.getRotationMatrix2D(center,angle,scale)
    result = cv2.warpAffine(img_c,M,(cols,rows))
    
    return result


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(img, table)
    
    return result


def flip_image(image):
    return np.flip(image, axis=1)


def read_img_rgb(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_img_rgb(image_path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)
    print('>>', image_path)

    
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def process_data(data_folder1, data_folder):
    make_dir(os.path.join(data_folder1, 'image_2'))
    make_dir(os.path.join(data_folder1, 'gt_image_2'))
    
    images_jtr = glob(os.path.join(data_folder1, 'image_2', 'jtr_*.png'))
    if (len(images_jtr)==0):
        print(">> Processing data...")
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        images = []
        gt_images = []
        
        print('>> Generate the jittered data set...')
        for image_file in image_paths:
            gt_image_file = label_paths[os.path.basename(image_file)]
            print(image_file)
            # read images (RGB)
            image = read_img_rgb(image_file)
            gt_image = read_img_rgb(gt_image_file)
            
            # >>> Itself...
            images.append(np.copy(image))
            gt_images.append(np.copy(gt_image))
            # >>> Scale up...
            images.append(transform_scale(image, ratio=1.25))
            gt_images.append(transform_scale(gt_image, ratio=1.25))
            # >>> Rotate CCW...
            images.append(transform_rotate(image, angle=10, scale=1.5))
            gt_images.append(transform_rotate(gt_image, angle=10, scale=1.5))
            # >>> Rotate CW...
            images.append(transform_rotate(image, angle=-10, scale=1.5))
            gt_images.append(transform_rotate(gt_image, angle=-10, scale=1.5))
            # >>> Flip ...
            images.append(flip_image(image))
            gt_images.append(flip_image(gt_image))
            
        # >> Adjust gamma...
        images_jtr = [adjust_gamma(img, gamma=1.2) for img in images]
        
        # >> Save images...
        print('>> Save images...')
        prefix = 'jtr'
        for idx in range(len(images_jtr)):
            name_img = "{0}_{1:0>6d}.png".format(prefix, idx)# jtr_000001.png
            name_gt_img = "{0}_road_{1:0>6d}.png".format(prefix, idx)# jtr_road_000001.png
            
            path_img = os.path.join(data_folder1, 'image_2',name_img)
            path_gt_img = os.path.join(data_folder1, 'gt_image_2',name_gt_img)
            
            save_img_rgb(path_img, images_jtr[idx])
            save_img_rgb(path_gt_img, gt_images[idx])
            
        print('>> Done!\n')


def save_array_to_csv(data, directory, header):
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(os.path.join(directory, "{0}_{1}.csv".format(header, time.time())), data, delimiter=",", header=header)
        
        
def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

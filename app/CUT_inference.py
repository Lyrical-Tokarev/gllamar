"""Script to launch model in inference mode and compute data for given input image.

I plan to add celery or make it more effective in future, but for now it's just a console script.
This is based on `test.py` script from original CUT repo.

sample launch

```
python CUT_inference.py <params the same as in train> <>
```
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

from .base_options import BaseOptions

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleImageDataset(BaseDataset):
    """This dataset class can load a set of 1 images specified by the path --input /path/to/file.
    Code is based on on SingleDataset

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if not os.path.exists(opt.input):
            return None
        if os.path.isdir(opt.input):
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        else:
            self.A_paths = []
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


class InferenceOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saved images go here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', default=True, help='use eval mode during test time.')
        parser.add_argument('--input', type=str, help='file name to run')
        parser.add_argument('--output', type=str, help='file name to save')
        #parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = "single"
    opt.num_test = 1
    dataset = SingleImageDataset(opt) #create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = SingleImageDataset(util.copyconf(opt, phase="test")) #create_dataset(util.copyconf(opt, phase="test"))
    if dataset is None:
        print("Path to file is incorrect, exiting")
        exit(1)
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    output_path = self.output #os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    #print('creating web directory', web_dir)
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    assert len(dataset) == 1
    # for now we only support single image as an input

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        # img_path is not used now, plan to use it with message queues
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        print(visuals)
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data)
            util.save_image(im, output_path, aspect_ratio=aspect_ratio)
        #save_images(webpage, visuals, img_path, width=opt.display_winsize)
    #webpage.save()  # save the HTML

import ntpath
import os
import time
from io import BytesIO  # Python 3.x

import scipy.misc

from . import html, util


class Visualizer:
    def __init__(self, opt, rank=0):
        self.rank = rank
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf

            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, "logs")
            if self.rank == 0:
                self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            if self.rank == 0:
                print("create web directory %s..." % self.web_dir)
                util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(
                opt.checkpoints_dir, opt.name, "loss_log.txt"
            )
            if self.rank == 0:
                with open(self.log_name, "a") as log_file:
                    now = time.strftime("%c")
                    log_file.write(
                        "================ Training Loss (%s) ================\n"
                        % now
                    )

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if 0:  # do not show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                if self.rank == 0:
                    scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                    # Create an Image object
                    img_sum = self.tf.Summary.Image(
                        encoded_image_string=s.getvalue(),
                        height=image_numpy.shape[0],
                        width=image_numpy.shape[1],
                    )
                    # Create a Summary value
                    img_summaries.append(
                        self.tf.Summary.Value(tag=label, image=img_sum)
                    )

            if self.rank == 0:
                # Create and write Summary
                summary = self.tf.Summary(value=img_summaries)
                self.writer.add_summary(summary, step)

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(
                            self.img_dir,
                            "epoch%.3d_%s_%d.png" % (epoch, label, i),
                        )
                        if self.rank == 0:
                            util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(
                        self.img_dir, "epoch%.3d_%s.png" % (epoch, label)
                    )
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    if self.rank == 0:
                        util.save_image(image_numpy, img_path)

            if self.rank == 0:
                # update website
                webpage = html.HTML(
                    self.web_dir, "Experiment name = %s" % self.name, refresh=5
                )
                for n in range(epoch, 0, -1):
                    webpage.add_header("epoch [%d]" % n)
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():
                        if isinstance(image_numpy, list):
                            for i in range(len(image_numpy)):
                                img_path = "epoch%.3d_%s_%d.png" % (n, label, i)
                                ims.append(img_path)
                                txts.append(label + str(i))
                                links.append(img_path)
                        else:
                            img_path = "epoch%.3d_%s.png" % (n, label)
                            ims.append(img_path)
                            txts.append(label)
                            links.append(img_path)
                    if len(ims) < 10:
                        webpage.add_images(
                            ims, txts, links, width=self.win_size
                        )
                    else:
                        num = int(round(len(ims) / 2.0))
                        webpage.add_images(
                            ims[:num],
                            txts[:num],
                            links[:num],
                            width=self.win_size,
                        )
                        webpage.add_images(
                            ims[num:],
                            txts[num:],
                            links[num:],
                            width=self.win_size,
                        )
                webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                if self.rank == 0:
                    summary = self.tf.Summary(
                        value=[
                            self.tf.Summary.Value(tag=tag, simple_value=value)
                        ]
                    )
                    self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, mt):
        message = "(epoch: %d, iters: %d, time: %.3f, model_time: %.3f) " % (
            epoch,
            i,
            t,
            mt,
        )
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += "%s: %.3f " % (k, v)

        if self.rank == 0:
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write("%s\n" % message)

    def convert_visuals_to_numpy(self, visuals, gray=False):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 1
            if "input_label" == key and not gray:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            elif "input_label" == key and gray:
                t = util.tensor2labelgray(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, gray=False):
        visuals = self.convert_visuals_to_numpy(visuals, gray=gray)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        if self.rank == 0:
            webpage.add_header(name)
        ims = []
        txts = []
        links = []

        cnt = 0
        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, "%s.png" % (name))
            save_path = os.path.join(image_dir, image_name)
            if self.rank == 0:
                util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
            cnt += 1
            if cnt % 4 == 0:
                if self.rank == 0:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                ims = []
                txts = []
                links = []

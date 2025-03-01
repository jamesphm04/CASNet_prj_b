from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
from models import *
from utils import *
from loss_functions import *
from dataset import get_dataset


def set_converts(datasets):
    '''
    Generates conversion pairs for image-to-image translation training,
    testing, and tensorboard visualization for classification tasks with
    exactly two datasets.

    Examples:

    1. MNIST <-> MNIST-M
        train_converts = ['M2MM', 'MM2M']
        test_converts = ['M2MM', 'MM2M']
        tensorboard_converts = ['M2MM', 'MM2M']

    2. SynthCan <-> PhysCan
        train_converts = ['SC2PC', 'PC2SC']
        test_converts = ['SC2PC', 'PC2SC']
        tensorboard_converts = ['SC2PC', 'PC2SC']
    '''

    if len(datasets) != 2:
        raise Exception("This function only supports exactly two datasets.")

    dset1, dset2 = datasets
    train_converts = [f'{dset1}2{dset2}', f'{dset2}2{dset1}']
    test_converts = train_converts.copy()
    tensorboard_converts = train_converts.copy()

    return train_converts, test_converts, tensorboard_converts


class Trainer:
    def __init__(self, args):
        self.args = args
        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets)
        self.imsize = (args.imsize, args.imsize)
        self.acc = dict()

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.args.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.args.batch,
                                                    imsize=self.imsize, workers=self.args.workers)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions(args)

        self.writer = SummaryWriter('./tensorboard/%s' % args.ex)
        self.logger = getLogger()
        self.checkpoint = './checkpoint/%s' % (args.ex)
        self.step = 0

    def set_default(self):
        torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        torch.cuda.manual_seed_all(self.args.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        if not os.path.exists(self.checkpoint + '/%d' % self.step):
            os.mkdir(self.checkpoint + '/%d' % self.step)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, dset))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))

    def load_networks(self, step):
        self.step = step
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    self.nets[key][dset].load_state_dict(torch.load(self.checkpoint
                                                                    + '/%d/net%s_%s.pth' % (step, key, dset)))
            else:
                self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)))

    def set_networks(self):
        self.nets['E'] = Encoder()
        self.nets['G'] = Generator()
        self.nets['S'] = Separator(self.imsize, self.training_converts)
        self.nets['D'] = dict()
        for dset in self.args.datasets:
            if dset == 'MM':
                self.nets['D'][dset] = Discriminator_MNIST()
            elif dset == 'M':
                self.nets['D'][dset] = Discriminator_MNIST()
            else:
                # self.nets['D'][dset] = Discriminator_Can()
                self.nets['D'][dset] = PatchGAN_Discriminator()

        # initialization
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    init_params(self.nets[net][dset])
            else:
                init_params(self.nets[net])
        self.nets['P'] = VGG19()

        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].cuda()
            else:
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.args.lr_cas,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_cas)

        self.optims['D'] = dict()
        for dset in self.args.datasets:
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.args.lr_cas,
                                                betas=(self.args.beta1, 0.999),
                                                weight_decay=self.args.weight_decay_cas)

        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.args.lr_cas,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_cas)

        self.optims['S'] = optim.Adam(self.nets['S'].parameters(), lr=self.args.lr_cas,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_cas)

    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].train()
            else:
                self.nets[net].train()


    def get_batch(self, batch_data_iter):
        batch_data = dict()
        # Instead of using .next() method, use a for loop to iterate through the DataLoader
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(iter(batch_data_iter[dset]))
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(iter(batch_data_iter[dset]))
        return batch_data

    def train_dis(self, imgs):  # Train Discriminators (D)
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()

        # Real
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            # TODO check slices
            D_outputs_real[dset] = self.nets['D'][dset](imgs[dset])
            #D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))

        contents, styles = self.nets['S'](features, self.training_converts)

        # CADT
        if self.args.CADT:
            for convert in self.training_converts:
                source, target = convert.split('2')
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split('2')
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

            # Non patchGAN
            # D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])

            # patchGAN
            D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))

        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        for optimizer in self.optims['D'].values():
            optimizer.step()
        self.losses['D'] = errD.data.item()

    def train_esg(self, imgs):  # Train Encoder(E), Separator(S), Generator(G)
        self.set_zero_grad()
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted = dict()
        perceptual, style_gram = dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        con_sim = dict()
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            recon_imgs[dset] = self.nets['G'](features[dset], 0)
            perceptual[dset] = self.nets['P'](imgs[dset])
            style_gram[dset] = [gram(fmap) for fmap in perceptual[dset][:-1]]
        contents, styles = self.nets['S'](features, self.training_converts)
        for convert in self.training_converts:
            source, target = convert.split('2')
            if self.args.CADT:
                con_sim[convert], styles[target] = cadt(contents[source], contents[target], styles[target])
                style_gram[target] = cadt_gram(style_gram[target], con_sim[convert])
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            # TODO check slice patches
            D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            #D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
            features_converted[convert] = self.nets['E'](converted_imgs[convert])
            perceptual_converted[convert] = self.nets['P'](converted_imgs[convert])
            style_gram_converted[convert] = [gram(fmap) for fmap in perceptual_converted[convert][:-1]]
        contents_converted, styles_converted = self.nets['S'](features_converted)

        Content_loss = self.loss_fns.content_perceptual(perceptual, perceptual_converted)

        # SWD style loss test
        # Style_loss = self.loss_fns.discrepancy_slice_wasserstein_style_loss(perceptual, perceptual_converted)

        # GRAM style loss
        Style_loss = self.loss_fns.style_perceptual(style_gram, style_gram_converted)

        Consistency_loss = self.loss_fns.consistency(contents, styles, contents_converted, styles_converted,
                                                     self.training_converts)
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, recon_imgs)

        errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss

        errESG.backward()
        for net in ['E', 'S', 'G']:
            self.optims[net].step()

        self.losses['G'] = G_loss.data.item()
        self.losses['Recon'] = Recon_loss.data.item()
        self.losses['Consis'] = Consistency_loss.data.item()
        self.losses['Content'] = Content_loss.data.item()
        self.losses['Style'] = Style_loss.data.item()

    def save_images_if_needed(self, imgs, labels, threshold):
        for convert in self.training_converts:
            convert_dir = os.path.join("data/converted", convert)
            if not os.path.exists(convert_dir):
                os.makedirs(convert_dir)  # Create the directory if it doesn't exist
            num_images = len([name for name in os.listdir(convert_dir) if name.endswith('.png')])
            if num_images < threshold:
                self.save_images(imgs, labels)
            else:
                print("All images generated")
                exit(0)

    def save_images(self, imgs, labels):
        features, converted_imgs, recon_imgs = dict(), dict(), dict()
        converts = self.tensorboard_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                recon_imgs[dset] = self.nets['G'](features[dset], 0)
            contents, styles = self.nets['S'](features, self.training_converts)
            for convert in self.training_converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            # 3 datasets
            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split('2')
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])

        for convert in converts:
            source, target = convert.split('2')
            # Create directory to save converted images if it doesn't exist
            convert_dir = os.path.join("data/converted", convert)
            if not os.path.exists(convert_dir):
                os.makedirs(convert_dir)

            # Find the maximum index of images already present in the directory
            convert_dir_content = os.listdir(convert_dir)
            max_index = 0
            for filename in convert_dir_content:
                if filename.endswith('.png'):
                    index = int(filename.split('_')[0])
                    max_index = max(max_index, index)
            # Save each converted image with an increasing index
            for i, img_tensor in enumerate(converted_imgs[convert]):
                img_filename = f"{max_index + i + 1}_class_{labels[source][i]}.png"
                img_path = os.path.join(convert_dir, img_filename)
                vutils.save_image(img_tensor, img_path, normalize=True, scale_each=True)
        for convert in converts:
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=True, nrow=8)
            img_filename = f"{convert}_converted_images.png"
            img_path = os.path.join("data/converted", img_filename)
            vutils.save_image(x, img_path)

    def tensor_board_log(self, imgs, labels):
        nrow = 8
        features, converted_imgs, recon_imgs = dict(), dict(), dict()
        converts = self.tensorboard_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                recon_imgs[dset] = self.nets['G'](features[dset], 0)
            contents, styles = self.nets['S'](features, self.training_converts)
            for convert in self.training_converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

        # Input Images & Reconstructed Images
        for dset in self.args.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
            x = vutils.make_grid(recon_imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Recon_Images/%s' % dset, x, self.step)

        # Converted Images
        for convert in converts:
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

    def print_loss(self):
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|' % (key, self.losses[key]))
        self.logger.info(
            '[%d/%d] %s| %s'
            % (self.step, self.args.iter, losses, self.args.ex))


    def train(self):
        self.set_default()
        self.set_networks()
        if self.args.resume_checkpoint:
            self.load_networks(self.args.load_step)
            self.step = self.args.load_step
        self.set_optimizers()
        self.logger.info(self.loss_fns.alpha)
        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.args.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            # training
            for u in range(1):
                self.train_dis(imgs) # Train Discriminator(D)
            for t in range(2):
                self.train_esg(imgs) # Train Encoder(E), Separator(S), Generator(G)

            # # tensorboard
            # if self.step % self.args.tensor_freq == 0:
            #     self.tensor_board_log(imgs, labels)

            # Save images
            if self.args.gen_data:
                if self.step >= self.args.save_step:
                    self.save_images_if_needed(imgs, labels, self.args.num_imgs)

            # save network step
            if self.step % self.args.net_freq == 0:
                self.save_networks()
            self.print_loss()

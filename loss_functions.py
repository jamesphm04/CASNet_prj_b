import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

def loss_weights(dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'], alpha['gradient_penalty'] = dict(), dict(), dict(), dict()
    alpha['recon'], alpha['consis'], alpha['content']= 5, 1, 2 #5, 1, 1

    # Synthetic Cans <-> Physical Cans
    if 'SC' in dsets and 'PC' in dsets and 'U' not in dsets:
        alpha['style']['SC2PC'], alpha['style']['PC2SC'] = 4e4, 4e4 # 1e4, 1e4
        alpha['dis']['SC'], alpha['dis']['PC'] = 1, 1 #1, 1
        alpha['gradient_penalty']['SC'], alpha['gradient_penalty']['PC'] = 0.5, 0.5 #0.5,0.5
        alpha['gen']['SC'], alpha['gen']['PC'] = 1, 1 #1, 1

    # MNIST <-> MNIST-M
    if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
        alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4#1, 0.2#5e4, 1e4
        alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
        alpha['gradient_penalty']['M'], alpha['gradient_penalty']['MM'] = 1, 1
        alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

    # MNIST <-> USPS
    elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
        alpha['style']['M2U'], alpha['style']['U2M'] = 0.75, 0.75
        alpha['dis']['M'], alpha['dis']['U'] = 1, 1
        alpha['gradient_penalty']['M'], alpha['gradient_penalty']['U'] = 3, 3
        alpha['gen']['M'], alpha['gen']['U'] = 1.2, 1.2

    # MNIST <-> MNIST-M <-> USPS
    elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
        alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
        alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
        alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5


    return alpha

class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.datasets)

    def compute_gradient_penalty(self, real, fake, discriminators):
        gradient_penalty = 0

        for cv in fake.keys():
            source, target = cv.split('2')
            # Generate random weights for interpolation
            alpha = torch.rand(real[target].size(0), 1, 1, 1).to(real[target].device)

            # Create interpolated samples between real and fake
            interpolates = (alpha * real[target] + (1 - alpha) * fake[cv]).requires_grad_(True)
            # Calculate discriminator output for interpolated samples
            disc_interpolates = discriminators[target](interpolates)

            # Compute gradients of the output with respect to the inputs
            gradients = torch_grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(real[target].device),
                                      create_graph=True, retain_graph=True)[0]
            gradients = gradients.view(real[target].size()[0], -1)

            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
            # Compute gradient penalty for each dataset
            gradient_penalty += self.alpha['gradient_penalty'][target] * ((gradients_norm - 1) ** 2).mean()


        return gradient_penalty

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        # WGAN loss
        #for dset in real.keys():
        #    dis_loss += self.alpha['dis'][dset] * (-real[dset]).mean()
        #for cv in fake.keys():
        #    source, target = cv.split('2')
        #    dis_loss += self.alpha['dis'][target] * fake[cv].mean()

        # DCGAN loss
        #for dset in real.keys():
        #    dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
        #for cv in fake.keys():
        #    source, target = cv.split('2')
        #    dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))

        # Hinge loss
        for dset in real.keys():
            dis_loss += self.alpha['dis'][dset] * F.relu(1 - real[dset]).mean()
        for cv in fake.keys():
            source, target = cv.split('2')
            dis_loss += self.alpha['dis'][target] * F.relu(1 + fake[cv]).mean()
        return dis_loss


    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            #gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            gen_loss += -self.alpha['gen'][target] * fake[cv].mean()

        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def discrepancy_slice_wasserstein_style_loss(self, perceptual, perceptual_converted, projection_dimension=64):
        swd_loss=0
        style_loss=0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            for p1, p2 in zip(perceptual[target][:-1], perceptual_converted[cv][:-1]):
                s = p1.shape
                if s[0] > 1:
                    proj = torch.randn(s[2], projection_dimension, device="cuda:0")
                    proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
                    p1 = torch.matmul(p1, proj)
                    p2 = torch.matmul(p2, proj)
                p1 = torch.topk(p1, s[0], dim=0)[0]
                p2 = torch.topk(p2, s[0], dim=0)[0]
                dist = p1 - p2
                wdist = torch.mean(torch.mul(dist, dist))
                swd_loss += wdist
            style_loss +=self.alpha['style'][cv]*swd_loss
        return style_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss


    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss


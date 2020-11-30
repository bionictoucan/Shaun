import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from .losses import PerceptualLoss
import os, h5py
from .shaun import Shaun
from time import time
from crispy.utils import pt_vibrant

class Trainer:
    """
    The class used to set up the training and do the training and save the models for the network.

    Parameters
    ----------
    in_channels : int
        The number of channels of the images input to the network.
    out_channels : int
        The number of channels of the output images.
    nef : int
        The base number of feature maps to use at the first convolutional layer.
    data_pth : str
        The path to the training data.
    slic_pth : str
        The path to the Slic repository.
    slic_model_pth : str
        The path to the trained Slic model.
    save_dir : str
        The directory to save the models in.
    minibatches_per_epoch : int
        The number of minibatches to train on per epoch.
    max_val_batches : int
        The number of batches to do the validation on.
    layer : int
        The layer that the feature maps are compared at in Slic.
    """
    def __init__(self, in_channels, out_channels, nef, data_pth, slic_pth, slic_model_pth, save_dir, minibatches_per_epoch, max_val_batches, layer):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Shaun(in_channels=in_channels, out_channels=out_channels, nef=nef).to(self.device)
        print("=> Created Shaun.")
        self.data_pth = data_pth
        self.perceptual_loss = PerceptualLoss(loss=nn.MSELoss(), slic_pth=slic_pth, model_pth=slic_model_pth, layer=layer)
        print("=> Shaun has become perceptually aware.")
        self.mse_loss = nn.MSELoss()
        print("=> Shaun knows how to calculate distances.")
        self.combined_loss = lambda *x: (self.perceptual_loss.get_loss(*x), self.mse_loss(*x))
        self.save_dir = save_dir
        self.current_epoch = 0
        self.minibatches_per_epoch = minibatches_per_epoch
        self.max_val_batches = max_val_batches

    def load_data(self):
        """
        This class method loads the training data and splits it into training and validation data 90/10%.
        """
        f = h5py.File(self.data_pth, "r")
        inp, out = np.array(f.get("input")), np.array(f.get("output"))
        indices = np.arange(inp.shape[0])
        np.random.RandomState(seed=42).shuffle(indices)
        max_idx = int(0.9*indices.shape[0])

        self.train_in, self.train_out = inp[indices[:max_idx]], out[indices[:max_idx]]
        self.val_in, self.val_out = inp[indices[max_idx:]], out[indices[max_idx:]]

    def checkpoint(self):
        """
        This class method creates a checkpoint for the current epoch.
        """
        if hasattr(self, "scheduler"):
            self.chkpt = {
                "epoch" : self.current_epoch,
                "model_state_dict" : self.model.state_dict(),
                "optimiser_state_dict" : self.optimiser.state_dict(),
                "scheduler_state_dict" : self.scheduler.state_dict(),
                "losses" : self.losses
            }
        else:
            self.chkpt = {
                "epoch" : self.current_epoch,
                "model_state_dict" : self.model.state_dict(),
                "optimiser_state_dict" : self.optimiser.state_dict(),
                "losses" : self.losses
            }

    def save_checkpoint(self):
        """
        This class method saves the current checkpoint.
        """
        save_pth = f"{self.save_dir}{self.current_epoch}.pth"
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename):
        """
        This class method loads a checkpoint.

        Parameters
        ----------
        filename : str
            The checkpoint to be loaded.
        """
        if os.path.isfile(filename):
            print(f"=> loading checkpoint {filename}")
            ckp = torch.load(filename)
            self.current_epoch = ckp["epoch"]
            self.losses = ckp["losses"]
            self.model.load_state_dict(ckp["model_state_dict"])
            if hasattr(self, "optimiser"):
                self.optimiser.load_state_dict(ckp["optimiser_state_dict"])
            if hasattr(self, "scheduler"):
                self.scheduler.load_state_dict(ckp["scheduler_state_dict"])

            train_p = self.losses["train_p"][-1]
            train_l = self.losses["train_l"][-1]
            train_c = self.losses["train_c"][-1]
            val_p = self.losses["val_p"][-1]
            val_l = self.losses["val_l"][-1]
            val_c = self.losses["val_c"][-1]
            print(f"=> loaded checkpoint {filename} at epoch {self.current_epoch} with training perceptual {train_p}, training L2 {train_l}, training combined {train_c}, validation perceptual {val_p}, validation L2 {val_l}, validation combined {val_c}")
        else:
            print(f"=> no checkpoint found at {filename}")

    def train(self, train_loader, scheduler=False):
        """
        This class method carries out the training for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data.
        scheduler : bool, optional
            Whether or not there is a learning rate scheduler.
        """
        pt_losses, lt_losses, t_losses = [], [], []
        self.model.train()
        minibatch_idx = 0
        for jj, (blr, img) in enumerate(train_loader):
            minibatch_idx += 1
            if minibatch_idx > self.minibatches_per_epoch:
                break
            blr = blr.float().to(self.device).unsqueeze(1)
            img = img.float().to(self.device).unsqueeze(1)
            output = self.model(blr)
            self.optimiser.zero_grad()

            p_loss, mse_loss = self.combined_loss(output, img)
            loss = p_loss + mse_loss
            loss.backward()
            self.optimiser.step()

            pt_losses.append(p_loss.item())
            lt_losses.append(mse_loss.item())
            t_losses.append(loss.item())

            if jj == 0:
                plt_blr = blr[0,0].clone().detach().cpu().squeeze().numpy()
                plt_gen = output[0,0].clone().detach().cpu().squeeze().numpy()
                plt_img = img[0,0].clone().detach().cpu().squeeze().numpy()
        if scheduler:
            self.scheduler.step()

        return np.mean(np.array(pt_losses)), np.mean(np.array(lt_losses)), np.mean(np.array(t_losses)), plt_blr, plt_gen, plt_img

    def validation(self, val_loader):
        """
        This class method does the validation for one epoch.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The validation data.
        """
        pv_losses, lv_losses, v_losses = [], [], []
        self.model.eval()
        val_batches = 0
        with torch.no_grad():
            for jj, (blr, img) in enumerate(val_loader):
                val_batches += 1
                if val_batches > self.max_val_batches:
                    break
                blr = blr.float().to(self.device).unsqueeze(1)
                img = img.float().to(self.device).unsqueeze(1)

                output = self.model(blr)

                p_loss, mse_loss = self.combined_loss(output, img)
                loss = p_loss + mse_loss

                pv_losses.append(p_loss.item())
                lv_losses.append(mse_loss.item())
                v_losses.append(loss.item())

        return np.mean(np.array(pv_losses)), np.mean(np.array(lv_losses)), np.mean(np.array(v_losses))

    def arcade_star(self, num_epochs, lr, reg=1e-6, batch_size=64, load=False, load_pth=None, scheduler=False, n_oscillate=100, lr_min=1e-5):
        """
        This class method trains the network with the interactive plotting environment.

        Parameters
        ----------
        num_epochs : int
            The total number of epochs to train for.
        lr : float
            The learning rate of the system.
        reg : float, optional
            The regularisation parameter for the optimiser. Default is 1e-6.
        batch_size : int, optional
            The batch size of the data loaders. Default is 64.
        load : bool, optional
            Whether or not an earlier model is being restored. Default is False.
        load_pth : str, optional
            The path to the earlier model that is being restored. Default is None.
        scheduler : bool, optional
            Whether or not a learning rate scheduler is used. Default is False.
        n_oscillate : int, optional
            The number of epochs the learning rate is descreased before being reset to maximum. Default is 100.
        lr_min : float, optional
            The minimum learning rate that is scheduled. Default is 1e-5.
        """
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)

        if scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=n_oscillate, eta_min=lr_min)

        if load:
            print("=> Shaun has been restored to an earlier save.")
            self.load_checkpoint(load_pth)

        #=====================================================================#
        # dataset and data loader creation
        train_dataset = TensorDataset(torch.from_numpy(self.train_in), torch.from_numpy(self.train_out))
        val_dataset = TensorDataset(torch.from_numpy(self.val_in), torch.from_numpy(self.val_out))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        print("=> created data loaders")

        #=====================================================================#
        # initialisation of the plotting environment

        fig = plt.figure(figsize=(9,9), constrained_layout=True)
        gs = fig.add_gridspec(nrows=4, ncols=3)
        pt_ax = fig.add_subplot(gs[1,:])
        lt_ax = pt_ax.twinx()
        t_ax = fig.add_subplot(gs[2,:])
        v_ax = t_ax.twinx()
        pv_ax = fig.add_subplot(gs[3,:])
        lv_ax = pv_ax.twinx()
        bi_ax = fig.add_subplot(gs[:1,0])
        ci_ax = fig.add_subplot(gs[:1,1])
        ti_ax = fig.add_subplot(gs[:1,2])
        pt_ax.set_ylabel("Perceptual loss", color=pt_vibrant["teal"])
        lt_ax.set_ylabel("L2 loss", color=pt_vibrant["red"])
        t_ax.set_ylabel("Training loss", color=pt_vibrant["cyan"])
        v_ax.set_ylabel("Validation loss", color=pt_vibrant["orange"])
        pv_ax.set_ylabel("Perceptual loss", color=pt_vibrant["teal"])
        lv_ax.set_ylabel("L2 loss", color=pt_vibrant["red"])
        bi_ax.set_xticks([])
        bi_ax.set_yticks([])
        ci_ax.set_xticks([])
        ci_ax.set_yticks([])
        ti_ax.set_xticks([])
        ti_ax.set_yticks([])
        pt_ax.grid(True)
        t_ax.grid(True)
        pv_ax.grid(True)
        print("=> the interactive plotting environment has been created")
        fig.show()
        fig.canvas.draw()

        #=====================================================================#
        # define lists to store the different losses in

        perct_losses, mset_losses, train_losses, percv_losses, msev_losses, val_losses = [], [], [], [], [], []

        #=====================================================================#
        # do the training and validation

        t_init = time()
        for j in range(num_epochs):
            if j != 0:
                self.current_epoch += 1
            if j == 0 and load:
                self.current_epoch += 1

            pt, lt, t, plt_blr, plt_gen, plt_img = self.train(train_loader, scheduler=scheduler)
            perct_losses.append(pt)
            mset_losses.append(lt)
            train_losses.append(t)

            pv, lv, v = self.validation(val_loader)
            percv_losses.append(pv)
            msev_losses.append(lv)
            val_losses.append(v)
            t_now = round(time() - t_init, 3)

        #======================================================================
        # save the model
            self.losses = {
                "train_p" : perct_losses,
                "train_l" : mset_losses,
                "train_c" : train_losses,
                "val_p" : percv_losses,
                "val_l" : msev_losses,
                "val_c" : val_losses
            }

            self.checkpoint()
            self.save_checkpoint()

        #======================================================================
        # plot the results
            fig.suptitle(f"Time elapsed {t_now} s after epoch {self.current_epoch}")
            pt_ax.set_ylabel("Perceptual loss", color=pt_vibrant["teal"])
            lt_ax.set_ylabel("L2 loss", color=pt_vibrant["red"])
            pt_ax.semilogy(np.arange(j+1), perct_losses, color=pt_vibrant["teal"])
            lt_ax.semilogy(np.arange(j+1), mset_losses, color=pt_vibrant["red"])
            t_ax.set_ylabel("Training loss", color=pt_vibrant["cyan"])
            v_ax.set_ylabel("Validation loss", color=pt_vibrant["orange"])
            t_ax.semilogy(np.arange(j+1), train_losses, color=pt_vibrant["cyan"])
            v_ax.semilogy(np.arange(j+1), val_losses, color=pt_vibrant["orange"])
            pv_ax.set_ylabel("Perceptual loss", color=pt_vibrant["teal"])
            lv_ax.set_ylabel("L2 loss", color=pt_vibrant["red"])
            pv_ax.semilogy(np.arange(j+1), percv_losses, color=pt_vibrant["teal"])
            lv_ax.semilogy(np.arange(j+1), msev_losses, color=pt_vibrant["red"])
            bi_ax.imshow(plt_blr, cmap="Greys_r")
            ci_ax.imshow(plt_gen, cmap="Greys_r")
            ti_ax.imshow(plt_img, cmap="Greys_r")
            fig.canvas.draw()

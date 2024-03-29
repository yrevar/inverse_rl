import os
import sys
import PIL
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Torch
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
# ConvAE
import ConvAE as ConvAE

# GeoLife
sys.path.append("../dataset/GeolifeTrajectories1.3/")
import geolife_image_features as GLImgs

# Utils
sys.path.append("../utils_python/")
from PersistentDictionary import PersistentDictionary
import shutil
from pprint import pprint

CONFIG_FILE = None
EXP_FILE_BACKUP = True
exp_config_dict = dict(
    exp_title = "EXP_GEOLIFE_FEATURES_CONVAE_STRIDED_FINE_STATES_CHOPPED_TRAJ_RGB_DELETE",
    identifier_prefix="",
    identifier_suffix="",
    identifier_params=["states",
                       "img_size",
                       "conv_k_size",
                       "fc_latent_multiplier",
                       "code_size",
                       "strided_conv_freq",
                       "lr",
                       "weight_decay"],

    # data
    img_files_expr="../irl/mlirl/features/satellite_rgb/imgs_128x128/*.jpg",
    img_size="64x64",
    states="100x100",
    batch_size=32,
    drop_last=False,
#     input_dim=(1, 64, 64),
    input_dim=(3, 64, 64),
    MU=0,  # .2963,
    STD=1,  # 0.1986,

    # performance
    loss_criterion=nn.MSELoss(),
    acc_criterion=lambda a, b: ConvAE.pearson_corr_coeff(a, b),

    # training
    lr=0.0001,
    weight_decay=1e-9,
    optimizer_fn=lambda params, lr, weight_decay: torch.optim.Adam(
        params, lr=lr, weight_decay=weight_decay),
    tr_va_te_split=[0.6, 0.3, 0.1],
    n_epochs=30,

    # compute
    use_data_parallel=False,

    # arch
    nw_depth=6,
    conv_k_size=16,
    fc_latent_multiplier=4,
    code_size=128,
    strided_conv_freq=3,

    # results
    img_result_freq=5,
    print_loss_freq=0,
)
def compute_prediction_performance(imgs_loader, model, perf_fn, device):

    # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
    with torch.no_grad():
        perf = 0.
        for images in imgs_loader:
            images = images.view(-1, *input_dim)
            images = images.to(device)
            output = model(images)[0]
            perf += perf_fn(images, output).item()
    return perf / len(imgs_loader)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def convae_solver(
        model, input_dim, tr_loader, va_loader,
        device, num_epochs, optimizer_fn,
        loss_criterion, acc_criterion,
        title, results_dir=None, img_result_freq=5,
        print_loss_freq=10, verbose=False):

    # if verbose: print("{}".format(title))
    optimizer = optimizer_fn(model.parameters(), lr, weight_decay)

    loss_history = []
    train_acc_history = []
    val_acc_history = []
    n_tr_batches = len(tr_loader)

    # model.apply(weights_init)
    for epoch in range(num_epochs):

        # Epoch begin
        if verbose and print_loss_freq != 0:
            print("\t\t\t Loss [n_batch={}]: ".format(n_tr_batches), end="")
        total_loss = 0.
        for tr_batch_idx, images in enumerate(tr_loader):

            # Forward pass
            images = images.view(-1, *input_dim)
            images = Variable(images).to(device)
            output, code = model(images)

            # Compute loss
            loss = loss_criterion(output, images)
            total_loss += loss.item()
            loss_history.append(loss.item())
            if verbose and print_loss_freq and tr_batch_idx % print_loss_freq == 0:
                print(", {:.4f}".format(loss), end="")

            # Zero grads
            optimizer.zero_grad()
            # Caclulate grads (backprop)
            loss.backward()
            # Update weights
            optimizer.step()

            if epoch == 0 and tr_batch_idx == 0:
                train_acc = np.float64(compute_prediction_performance(
                    tr_loader, model, acc_criterion, device))
                val_acc = np.float64(compute_prediction_performance(
                    va_loader, model, acc_criterion, device))
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                if verbose:
                    msg = '\t\t Epoch [{}/{}]: train acc:{:.4f}, val acc:{:.4f}, loss: {:.4f}'.format(
                        epoch, num_epochs, train_acc, val_acc, loss)
                    print(msg)

        # Epoch end
        train_acc = compute_prediction_performance(tr_loader, model, acc_criterion, device)
        val_acc = compute_prediction_performance(va_loader, model, acc_criterion, device)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if verbose:
            msg = '\n\t\t Epoch [{}/{}]: train acc: {:.4f}, val acc: {:.4f}, loss: {:.4f}'.format(
                epoch + 1, num_epochs, train_acc, val_acc, total_loss / n_tr_batches)
            print(msg)

        if results_dir and img_result_freq and ((epoch+1) == 1 or (epoch+1) % img_result_freq == 0):

            images = iter(va_loader).next()
            images = images.view(-1, *input_dim)
            images = Variable(images).to(device)
            output, code = model(images)

            result = model.get_results_img(GLImgs.unnormalize_image(images, MU, STD),
                                           GLImgs.unnormalize_image(output, MU, STD))
            im = PIL.Image.fromarray(np.asarray(result))
            im.save(os.path.join(results_dir, 'image_{}.png'.format(epoch+1)))

            fig = plt.figure(figsize=(20, 18))
            gridspec.GridSpec(5, 2)
            plt.subplot2grid((5, 2), (0, 0), colspan=1, rowspan=5)
            
            if input_dim[0] == 1:
                plt.imshow(np.asarray(result), cmap="gray")
            else:
                plt.imshow(np.asarray(result))
                
            plt.title(msg)
            plt.xticks([]), plt.yticks([])

            plt.subplot2grid((5, 2), (0, 1), colspan=1, rowspan=1)
            plt.plot(train_acc_history, label="training")
            plt.plot(val_acc_history, label="validation")
            plt.ylabel("Pearson's Correlation Coefficient")
            plt.xlabel("Epoch")
            plt.legend()
            plt.title(
                title + "\n Best: train: {:.4f} val: {:.4f}".format(np.max(train_acc_history), np.max(val_acc_history)))

            plt.subplot2grid((5, 2), (3, 1), colspan=1, rowspan=1)
            plt.plot(loss_history)
            plt.ylabel("Loss")
            plt.xlabel("Batch ({})".format(batch_size))
            plt.title(
                title + "\n Best: train loss: {:.4f}".format(np.min(loss_history)))
            plt.savefig(os.path.join(results_dir, 'summary_ep_{}.png'.format(epoch+1)))
            plt.close()
            # display.clear_output(wait=True)
            # display.display(plt.gcf())
    return loss_history, train_acc_history, val_acc_history

def get_experiment_dirname(exp_config):
    
    os.makedirs(exp_config["exp_title"], exist_ok=True)
    
    exp_id = "".join("$" + str(param) + "=" + str(exp_config[param]) for param in exp_config["identifier_params"])[1:]
    if "identifier_prefix" in exp_config and exp_config["identifier_prefix"]:
        exp_id = exp_config["identifier_prefix"] + "$" + exp_id
    if "identifier_suffix" in exp_config and exp_config["identifier_suffix"]:
        exp_id += "$" + exp_config["identifier_suffix"]
        
    return os.path.join(exp_config["exp_title"], exp_id), exp_config["exp_title"] + "__" + exp_id
                        
if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    
    # Experiment configuration
    if CONFIG_FILE:
        
        EXP_CONFIG = PersistentDictionary(CONFIG_FILE, True)
        EXP_DIR, EXP_ID = get_experiment_dirname(EXP_CONFIG)
    else:   
        EXP_CONFIG = PersistentDictionary(None, verbose=True, **exp_config_dict)
        EXP_DIR, EXP_ID = get_experiment_dirname(EXP_CONFIG)
        EXP_CONFIG.write(os.path.join(EXP_DIR, "configs.p"))
        
    # Bring config parameters into scope
    globals().update(EXP_CONFIG)
    
    # Results
    RESULTS_DIR = os.path.join(EXP_DIR, "results")
    RESULTS_FILE = os.path.join(RESULTS_DIR, "results.p")
    RESULTS = PersistentDictionary(RESULTS_FILE, True).write()
    
    if EXP_FILE_BACKUP:
        exp_repro_file = "./{}/{}.py".format(exp_title, exp_title)
        os.makedirs(os.path.dirname(exp_repro_file), exist_ok=True)
        shutil.copy(__file__, exp_repro_file)
    
    title_line = "{}".format(EXP_ID).replace("__", ": ").replace("$", ", ")
    title = title_line.replace(": ", ":\n\t").replace(", ", ",\n\t")
    print("{}\n{}\n{}".format("="*80, title, "="*80))
    
    # Architecture
    print("Preparing Net..")
    CONV_BLOCK = [("conv1", conv_k_size), ("relu1", None)]
    CONV_LAYERS = ConvAE.create_network(CONV_BLOCK, nw_depth, pooling_freq=1e100,
                                        strided_conv_freq=strided_conv_freq, strided_conv_channels=conv_k_size)
    CONV_NW = CONV_LAYERS + [("flatten1", None),
                             ("linear1", fc_latent_multiplier * code_size), ("linear1", code_size), ]
    
    print("NW Config (depth={}):\n\tBlock: ".format(len(CONV_NW)), end="")
    pprint(CONV_BLOCK)
    print("\tNet: \n", end="")
    pprint(CONV_NW)
    
    print("Preparing Data..")
    IMG_FILES = img_files_expr #(states, img_size)
    # dataset = GLImgs.ImageFolderDataset(IMG_FILES, transform=transforms.Compose([
    #                 lambda x: np.asarray(x.convert("L"), np.uint8),
    #                 lambda x: torch.from_numpy(x).float().div(255.)]))
    # MU, STD = GLImgs.get_img_normalization_params(dataset)
    tr_loader, va_loader, te_loader = GLImgs.data_loader(
        IMG_FILES, batch_size=batch_size, drop_last=drop_last, tr_va_te_split=tr_va_te_split,
        transform=transforms.Compose([
#             lambda x: np.asarray(x.convert("L"), np.uint8)[32:-32,32:-32,np.newaxis],
            lambda x: np.asarray(x, np.uint8)[32:-32,32:-32,:],
            lambda x: torch.from_numpy(x).float().div(255.),
            lambda x: x.permute(2,0,1),
            lambda x: GLImgs.normalize_image(x, MU, STD)]))
    
    print("Creating model..")
    # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
    # torch.cuda.empty_cache() # Needed for repeated experiments
    model = ConvAE.ConvAE(input_dim, enc_config=CONV_NW)
    if use_data_parallel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))
    model.to(device)
    
    print("Training..")
    loss_history, train_acc_history, val_acc_history = convae_solver(
        model, input_dim, tr_loader, va_loader,
        device, n_epochs, optimizer_fn,
        loss_criterion, acc_criterion,
        title, RESULTS_DIR, img_result_freq=img_result_freq,
        print_loss_freq=print_loss_freq,
        verbose=True)
    
    print("Writing results..")
    RESULTS["loss_training"] = np.asarray(loss_history)
    RESULTS["acc_training"] = np.asarray(train_acc_history)
    RESULTS["acc_validation"] = np.asarray(val_acc_history)
    RESULTS.write()
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR,'./model_state_ae.pt'))
    torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR,'./model_state_encoder.pt'))
    torch.save(model.decoder.state_dict(), os.path.join(RESULTS_DIR,'./model_state_decoder.pt'))

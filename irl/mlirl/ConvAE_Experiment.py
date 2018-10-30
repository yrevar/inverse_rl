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
sys.path.append("../../dataset/GeolifeTrajectories1.3/")
import geolife_image_features as GLImgs

# Utils
sys.path.append("../../utils/")
from persistence import PickleWrapper
from persistence import ExperimentHelper
import shutil

exp_config = dict(
    
    IMG_FILES = "./features/state_100x100_features/imgs_64x64/*.jpg",
    
    batch_size = 128,
    drop_last = False,
    loss_criterion = nn.MSELoss(),
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, weight_decay=1e-7),
    acc_criterion = lambda a,b: -(nn.MSELoss()(a,b)),
    #lambda a,b: ConvAE.std_score(a, b, 0.1986, 4), #lambda a,b: 1.-(nn.MSELoss()(a,b))**(1./4),#lambda a,b: ConvAE.accuracy(a,b),
    use_data_parallel = False,
    input_dim = (1, 64, 64),
    tr_va_te_split = [0.6, 0.3, 0.1],
    
    conv_block = [("conv1", 16), ("relu1", None)],
    #penultimate_fc_size = lambda code_size: code_size * 4, 
    latent_multiplier_range = [1, 2, 4, 8],
    n_epochs = 20,
    n_repetitions = 3,
    code_size_range = [8, 16, 32, 64, 128, 256, 512], #, 2048, 4096, 8092],
    k_size_range = [16],
    nw_depth_range = range(1,10,1),
    img_result_freq = 5,
    print_loss_freq = 0,
    MU = 0.2963,
    STD = 0.1986,
)

def compute_prediction_performance(imgs_loader, model, perf_fn, device):
    
    # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
    with torch.no_grad():
        perf = 0.
        for images in imgs_loader:
            images = images.view(-1, *input_dim)
            # images = Variable(images)
            images = images.to(device)
            output = model(images)[0]
#             perf += perf_fn(ConvAE.normalize_0_1(images), ConvAE.normalize_0_1(output)).item()
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
    optimizer = optimizer_fn(model.parameters())
    
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
                train_acc = np.float64(compute_prediction_performance(tr_loader, model, acc_criterion, device))
                val_acc = np.float64(compute_prediction_performance(va_loader, model, acc_criterion, device))
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
            im = PIL.Image.fromarray(result)
            im.save(os.path.join(results_dir, 'image_{}.png'.format(epoch+1)))
            
            fig = plt.figure(figsize=(20, 18))
            gridspec.GridSpec(5,2)
            plt.subplot2grid((5,2), (0,0), colspan=1, rowspan=5)
            plt.imshow(result.transpose(1,0,2), cmap="gray")
            plt.title(msg)
            plt.xticks([]), plt.yticks([])

            plt.subplot2grid((5,2), (0,1), colspan=1, rowspan=1)
            plt.plot(train_acc_history, label="training")
            plt.plot(train_acc_history, label="validation")
            plt.ylabel("Accuracy (1 - Mean Abs Difference)")
            plt.xlabel("Epoch")
            plt.legend()
            plt.title(title)

            plt.subplot2grid((5,2), (3,1), colspan=1, rowspan=1)
            plt.plot(loss_history)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.title(title)
            plt.savefig(os.path.join(results_dir, 'summary_ep_{}.png'.format(epoch+1)))
            plt.close()
            # display.clear_output(wait=True)
            # display.display(plt.gcf())
    return loss_history, train_acc_history, val_acc_history

if __name__== "__main__":
    
    np.random.seed(0)
    torch.manual_seed(0)

    EXPERIMENT = "EXP_TEST_1"
    EXP_DATA_FILE = "./{}/exp_data.p".format(EXPERIMENT)
    EXP_REPRO_FILE = "./{}/{}.py".format(EXPERIMENT, EXPERIMENT)
    print("Launching Experiment: {} \n\t configuration file: {} \n\t reproduction file: {}".format(
        EXPERIMENT, EXP_DATA_FILE, EXP_REPRO_FILE))

    os.makedirs(os.path.dirname(EXP_REPRO_FILE), exist_ok=True)
    shutil.copy(__file__, EXP_REPRO_FILE)

    exp = ExperimentHelper(exp_config, EXP_DATA_FILE)
    exp.launch(globals())

    print("Preparing Data...")
    # dataset = GLImgs.ImageFolderDataset(IMG_FILES, transform=transforms.Compose([
    #                 lambda x: np.asarray(x.convert("L"), np.uint8),
    #                 lambda x: torch.from_numpy(x).float().div(255.)]))
    # MU, STD = GLImgs.get_img_normalization_params(dataset)

    tr_loader, va_loader, te_loader = GLImgs.data_loader(
        IMG_FILES, batch_size=batch_size, drop_last=drop_last, tr_va_te_split=tr_va_te_split, 
        transform=transforms.Compose([
            lambda x: np.asarray(x.convert("L"), np.uint8),
            lambda x: torch.from_numpy(x).float().div(255.),
            lambda x: GLImgs.normalize_image(x, MU, STD)]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    exp_tr_loss = exp.load("loss_training", default={})
    exp_tr_acc = exp.load("acc_training", default={})
    exp_va_acc = exp.load("acc_validation", default={})

    C = 0
    # T = len(k_size_range) * len(nw_depth_range)
    T = len(code_size_range) * len(nw_depth_range) * len(latent_multiplier_range)

    # code_size = code_size_range[0]
    k_size = k_size_range[0]

    for code_size in code_size_range:
        key1 = "code_{:04d}".format(code_size)

    # for k_size in k_size_range:
    #     key1 = "ksize_{:04d}".format(k_size)

        if key1 not in exp_tr_loss:
            exp_tr_loss[key1] = {}
            exp_tr_acc[key1] = {}
            exp_va_acc[key1] = {}

        conv_block = [("conv1", k_size), ("relu1", None)]

        for nw_depth in nw_depth_range:

            nw_conv = ConvAE.create_network(conv_block, nw_depth, pooling_freq=3)

            key2 = "deep_{:03d}".format(len(nw_conv) + 3)

            if key2 not in exp_tr_loss[key1]:
                exp_tr_loss[key1][key2] = {}
                exp_tr_acc[key1][key2] = {}
                exp_va_acc[key1][key2] = {}

            for latent_multiplier in latent_multiplier_range:

                C += 1

                key3 = "fc_latent_{:03d}".format(latent_multiplier)
                nw = nw_conv + [("flatten1", None), ("linear1", latent_multiplier * code_size), ("linear1", code_size),]
                p1, p2, p3 = code_size, len(nw), latent_multiplier

                title = "Code size: {}, Depth : {}, FC latent: {}".format(p1, p2, p3)

                if (key1 in exp_tr_loss) and (key2 in exp_tr_loss[key1]) and (key3 in exp_tr_loss[key1][key2]):
                    print("Previous results exist, skipping..")
                    continue

                for rep in range(n_repetitions):

                    print("[Rep {}/{}] {}/{}: {} \n\t Net: {}".format(rep+1, n_repetitions, C, T, title, nw))

                    # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
                    torch.cuda.empty_cache()

                    model = ConvAE.ConvAE(input_dim, enc_config=nw)        
                    if use_data_parallel and torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model = nn.DataParallel(model)
                    model.to(device)

                    if rep == 0:
                        results_dir = os.path.join(os.path.join(exp.data_dir, "results"), "code_{}_depth_{}_fc_latent{}".format(p1, p2, p3))
                        os.makedirs(results_dir, exist_ok=True)
                    else:
                        results_dir = None

                    loss_history, train_acc_history, val_acc_history = convae_solver(
                        model, input_dim, tr_loader, va_loader, 
                        device, n_epochs, optimizer_fn, 
                        loss_criterion, acc_criterion, 
                        title, results_dir, img_result_freq=img_result_freq, 
                        print_loss_freq=print_loss_freq,
                        verbose=True)

                    if rep == 0:
                        exp_tr_loss[key1][key2][key3] = np.asarray(loss_history)
                        exp_tr_acc[key1][key2][key3] = np.asarray(train_acc_history)
                        exp_va_acc[key1][key2][key3] = np.asarray(val_acc_history)
                    else:
                        exp_tr_loss[key1][key2][key3] += np.asarray(loss_history)
                        exp_tr_acc[key1][key2][key3] += np.asarray(train_acc_history)
                        exp_va_acc[key1][key2][key3] += np.asarray(val_acc_history)

                exp_tr_loss[key1][key2][key3] /= n_repetitions
                exp_tr_acc[key1][key2][key3] /= n_repetitions
                exp_va_acc[key1][key2][key3] /= n_repetitions

                print("Writing results to file..")
                exp.dump("loss_training", exp_tr_loss)
                exp.dump("acc_training", exp_tr_acc)
                exp.dump("acc_validation", exp_va_acc)

    # torch.save(model.state_dict(), './model_state.pt')

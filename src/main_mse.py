import datetime
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import argparse
import models
import losses
import time
# import wandb
import torch.utils.tensorboard

from torchvision import transforms
from util import AverageMeter, MAE, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from data import FeatureExtractor, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout

def parse_arguments():
    parser = argparse.ArgumentParser(description="Augmentation for multiview",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=50)

    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=200)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='cosine')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=None)

    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)

    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    parser.add_argument('--method', type=str, help='loss function', choices=['mae', 'mse'], default='mae')
    
    
    parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=False)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='none')

    parser.add_argument('--amp', action='store_true', help='use amp')

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True
    
    # If learning rate provided, it calculates the epochs where learning rate decay will occur.
    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    # It determines the warm-up strategy based on the decay method (cosine decay or step decay).
    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    opts.fairkl_kernel = opts.kernel != 'none'
    # opts.fairkl_kernel = opts.tf != 'none'
    # --- should this be opts.tf instead??
    return opts

# It applies different augmentations (crop, cutout, etc.) depending on the argument passed.
# It also normalizes the data and converts it to PyTorch tensors.
def get_transforms(opts):
    selector = FeatureExtractor("vbm")
    
    if opts.tf == 'none':
        aug = transforms.Lambda(lambda x: torch.tensor(x))

    elif opts.tf == 'crop':
        aug = transforms.Compose([
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])  

    elif opts.tf == 'cutout':
        aug = Cutout(patch_size=[1, 32, 32, 32], probability=0.5)

    elif opts.tf == 'all':
        aug = transforms.Compose([
            Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])
    
    T_pre = transforms.Lambda(lambda x: selector.transform(x))
    T_train = transforms.Compose([
        T_pre,
        aug,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    T_test = transforms.Compose([
        T_pre,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    return T_train, T_test



# def get_transforms(opts):
#     selector = FeatureExtractor("vbm")
    
#     # Define named functions to replace lambdas
#     def identity(x):
#         return x

#     def selector_transform(x):
#         return selector.transform(x)

#     def to_tensor(x):
#         return torch.from_numpy(x).float()

#     # Augmentation transformations
#     if opts.tf == 'none':
#         aug = transforms.Lambda(identity)

#     elif opts.tf == 'crop':
#         aug = transforms.Compose([
#             Crop((1, 121, 128, 121), type="random"),
#             Pad((1, 128, 128, 128))
#         ])  

#     elif opts.tf == 'cutout':
#         aug = Cutout(patch_size=[1, 32, 32, 32], probability=0.5)

#     elif opts.tf == 'all':
#         aug = transforms.Compose([
#             Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
#             Crop((1, 121, 128, 121), type="random"),
#             Pad((1, 128, 128, 128))
#         ])
    
#     # Transformations
#     T_pre = transforms.Lambda(selector_transform)
#     T_train = transforms.Compose([
#         T_pre,
#         aug,
#         transforms.Lambda(to_tensor),
#         transforms.Normalize(mean=0.0, std=1.0)
#     ])

#     T_test = transforms.Compose([
#         T_pre,
#         transforms.Lambda(to_tensor),
#         transforms.Normalize(mean=0.0, std=1.0)
#     ])

#     return T_train, T_test



# This function handles loading the dataset with transformations for training and testing.
# If train_all is set to True, it concatenates additional internal and external validation datasets.
def load_data(opts):
    # Gets the two sets of transforms
    T_train, T_test = get_transforms(opts)
    
    # internal = true - indicates that the internal data split is being used
    # transform = T_train -  Applies the transformations defined earlier for the training data (T_train)
    # load_feats = opts.biased_features - Optionally loads precomputed features if specified in opts.biased_features
    # train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train,
    #                         load_feats=opts.biased_features)
    train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train)


    # If so, it includes additional validation datasets in the training set
    if opts.train_all:
        valint_feats, valext_feats = None, None
        if opts.biased_features is not None:
            valint_feats = opts.biased_features.replace('.pth', '_valint.pth')
            valext_feats = opts.biased_features.replace('.pth', '_valext.pth')
        # Loads the internal and external validation datasets
        valint = OpenBHB(opts.data_dir, train=False, internal=True, transform=T_train,
                         load_feats=valint_feats)
        valext = OpenBHB(opts.data_dir, train=False, internal=False, transform=T_train,
                         load_feats=valext_feats)      
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, valint, valext])
        print("Total dataset lenght:", len(train_dataset))


    # Creates a PyTorch DataLoader for each set
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, 
                                               num_workers=8, persistent_workers=True)
   
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=8,
                                                persistent_workers=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=8,
                                                persistent_workers=True)

    return train_loader, test_internal, test_external

def load_model(opts):
    # If true, it initializes a model object for a ResNet-based architecture using the SupRegResNet class from the models module
    if 'resnet' in opts.model:
        model = models.SupRegResNet(opts.model)
    
    elif 'alexnet' in opts.model:
        model = models.SupRegAlexNet()
    
    elif 'densenet121' in opts.model:
        model = models.SupRegDenseNet()
    
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    # Moves the model to the device specified in opts.device. 
    # This can be either 'cuda' (for GPU) or 'cpu' (for CPU).
    model = model.to(opts.device)
    
    # A dictionary methods is defined to map loss function names (strings) to the corresponding PyTorch loss functions
    methods = {
        'mae': F.l1_loss,
        'mse': F.mse_loss
    }
    # This line selects the appropriate loss function from the methods dictionary based on the value of opts.method.
    regression_loss = methods[opts.method]

    return model, regression_loss

def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    return optimizer

# criterion = the loss function
def train(train_loader, model, criterion, optimizer, opts, epoch):
    # Initialize objects to track the average values for different metrics during training
    loss = AverageMeter()
    mae = MAE()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Sets up mixed-precision training if the opts.amp flag is set to True
    # Mixed-precision training uses 16-bit floating point numbers for certain operations to speed up training while using less memory
    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()

    # Records the current time to track how long it takes to process a batch of data
    t1 = time.time()
    # Loops over the batches of training data provided by train_loader. 
    # For each batch, images are the input data (e.g., images), and labels are the target values (e.g., ground truth values)
    # idx = current batch index 
    for idx, (images, labels, _) in enumerate(train_loader):
        # Updates the data_time meter with the time it took to load the current batch of data (calculated as the difference between the current time and t1)
        data_time.update(time.time() - t1)

        images, labels = images.to(opts.device), labels.to(opts.device)
        # stores the batch size for the current batch, which is the number of labels (or images) in the batch
        bsz = labels.shape[0]

        # Adjusts the learning rate during the early stages of training using a learning rate warmup strategy
        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        # Performs the forward pass and computes the loss
        # output is flattened (likely for regression or classification purposes)
        with torch.cuda.amp.autocast(scaler is not None):
            output, features = model(images)
            output = output.view(-1)
            running_loss = criterion(output, features, labels.float())
        
        optimizer.zero_grad()
        # backpropogation and optimization:
        # If mixed precision is not used (scaler is None), it performs standard backpropagation and updates the model parameters
        if scaler is None:
            running_loss.backward() 
            optimizer.step()
        # If mixed precision is enabled, the loss is scaled using scaler.scale(), and then backpropagation is done with scaler
        # The optimizer step is also done through the scaler, which ensures proper scaling of gradients in mixed precision
        else:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Updates the loss meter with the loss value for the current batch, using bsz (the batch size) to calculate the average
        loss.update(running_loss.item(), bsz)
        # Updates the MAE meter with the model's output and the true labels
        mae.update(output, labels)

        # Updates the batch_time meter with the time taken to process the current batch
        batch_time.update(time.time() - t1)
        # Estimates the remaining time for the current epoch by multiplying the average batch time by the number of remaining batches
        eta = batch_time.avg * (len(train_loader) - idx)

        # Print training progress periodically
        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t"
                  f"MAE {mae.avg:.3f}")

        # Update starting time for the next batch
        t1 = time.time()

    return loss.avg, mae.avg, batch_time.avg, data_time.avg

# ensures that the computation of gradients is disabled, which saves memory and computation since no backpropagation is needed during testing
@torch.no_grad()
# look at train() for further breakdown of each line
def test(test_loader, model, criterion, opts, epoch):
    loss = AverageMeter()
    mae = MAE()
    batch_time = AverageMeter()

    # puts the model in evaluation mode, which disables features like dropout and batch normalization that are only used during training
    model.eval()
    t1 = time.time()
    for idx, (images, labels, _) in enumerate(test_loader):
        images, labels = images.to(opts.device), labels.to(opts.device)
        bsz = labels.shape[0]

        output, features = model(images)
        output = output.view(-1)
        running_loss = criterion(output, features, labels.float())
        
        loss.update(running_loss.item(), bsz)
        mae.update(output, labels)

        batch_time.update(time.time() - t1)
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Test: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t"
                  f"MAE {mae.avg:.3f}")
    
        t1 = time.time()

    return loss.avg, mae.avg

if __name__ == '__main__':
    opts = parse_arguments()
    
    # ensures that the results are reproducible by setting a fixed random seed based on the opts.trial value. 
    # This helps control randomness in model initialization, data shuffling, and other random processes.
    set_seed(opts.trial)

    train_loader, test_loader_int, test_loader_ext = load_data(opts)
    model, criterion = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    # If opts.warm is True, it appends _warm to the model name, indicating a warm-up phase for the training.
    if opts.warm:
        model_name = f"{model_name}_warm"
    
    # This is used to name the experiment for logging or saving files.
    run_name = (f"{model_name}_{opts.method}_"
                f"{opts.optimizer}_"
                f"tf_{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"wd{opts.weight_decay}_"
                f"trainall_{opts.train_all}_"
                f"bsz{opts.batch_size}_"
                f"trial{opts.trial}")
    # Defines directories for saving TensorBoard logs (tb_dir) and model weights (save_dir) based on the opts.save_dir location
    # The ensure_dir function checks that these directories exist, and if not, it creates them
    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)
    save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    ensure_dir(tb_dir)
    ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.criterion = opts.method
    opts.optimizer_class = optimizer.__class__.__name__

    # This line initializes a run in Weights and Biases (W&B), a tool for tracking experiments, visualizing metrics, and sharing results
    # project = Specifies the name of the W&B project, which organizes runs under a specific project (in this case, "brain-age-prediction")
    # sync_tensorboard = Ensures that TensorBoard logs are synchronized with W&B so that you can visualize them on the W&B dashboard
    # tags = Adds a tag to the run, which can help categorize or filter the runs later in W&B
    # wandb.init(project="brain-age-prediction", config=opts, name=run_name, sync_tensorboard=True, tags=['to test'])
    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Criterion:', opts.criterion)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    # Initializes a TensorBoard writer (writer) to log scalar values (e.g., loss, MAE) and other metrics during training and testing. 
    # The logs will be saved to the tb_dir directory
    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    # Checks if Automatic Mixed Precision (AMP) is enabled (opts.amp)
    # AMP is used to speed up training and reduce memory consumption, especially when training on GPUs with Tensor cores
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, opts.epochs + 1):
        # Calls a function adjust_learning_rate to update the learning rate based on the epoch number
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss_train, mae_train, batch_time, data_time = train(train_loader, model, criterion, optimizer, opts, epoch)
        t2 = time.time()
        writer.add_scalar("train/loss", loss_train, epoch)
        # writer.add_scalar("train/mae", mae_train, epoch)

        # Tests the model on the internal test dataset
        # The test function returns the loss and MAE for each test set
        loss_test, mae_int = test(test_loader_int, model, criterion, opts, epoch)
        writer.add_scalar("test/loss_int", loss_test, epoch)
        # writer.add_scalar("test/mae_int", mae_int, epoch)

        # Tests the model on the external test dataset
        loss_test, mae_ext = test(test_loader_ext, model, criterion, opts, epoch)
        writer.add_scalar("test/loss_ext", loss_test, epoch)
        # writer.add_scalar("test/mae_ext", mae_ext, epoch)

        # Logs additional metrics like the learning rate (lr), batch time (BT), and data loading time (DT) to TensorBoard for further analysis
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_test:.4f} "
              f"mae_int {mae_int:.3f} mae_ext {mae_ext:.3f}")

        # Every opts.save_freq epochs, it computes additional metrics like MAE for age (compute_age_mae) and site BA (compute_site_ba), 
        # which are added to TensorBoard for further analysis
        if epoch % opts.save_freq == 0:
            # save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
            # save_model(model, optimizer, opts, epoch, save_file)
            mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader, test_loader_int, test_loader_ext, opts)
            
            writer.add_scalar("train/mae", mae_train, epoch)
            writer.add_scalar("test/mae_int", mae_int, epoch)
            writer.add_scalar("test/mae_ext", mae_ext, epoch)
            print("Age MAE:", mae_train, mae_int, mae_ext)

            ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader, test_loader_int, test_loader_ext, opts)
            writer.add_scalar("train/site_ba", ba_train, epoch)
            writer.add_scalar("test/ba_int", ba_int, epoch)
            writer.add_scalar("test/ba_ext", ba_ext, epoch)
            print("Site BA:", ba_train, ba_int, ba_ext)

            challenge_metric = ba_int**0.3 * mae_ext
            writer.add_scalar("test/score", challenge_metric, epoch)
            print("Challenge score", challenge_metric)

        # Saves the model weights (weights.pth) to the save_dir directory after each epoch
        # This allows the model to be restored later
        save_file = os.path.join(save_dir, f"weights.pth")
        save_model(model, optimizer, opts, epoch, save_file)
    
    # After training complete, work out final scores:
        
    mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader, test_loader_int, test_loader_ext, opts)
    writer.add_scalar("train/mae", mae_train, epoch)
    writer.add_scalar("test/mae_int", mae_int, epoch)
    writer.add_scalar("test/mae_ext", mae_ext, epoch)
    print("Age MAE:", mae_train, mae_int, mae_ext)

    ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader, test_loader_int, test_loader_ext, opts)
    writer.add_scalar("train/site_ba", ba_train, epoch)
    writer.add_scalar("test/ba_int", ba_int, epoch)
    writer.add_scalar("test/ba_ext", ba_ext, epoch)
    print("Site BA:", ba_train, ba_int, ba_ext)

    challenge_metric = ba_int**0.3 * mae_ext
    writer.add_scalar("test/score", challenge_metric, epoch)
    print("Challenge score", challenge_metric)
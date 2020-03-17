import logging
import argparse
import torch
import os
import helpers                                     # helpers for checkpointing, logging, etc
from net import Net                                # the neural net, metrics (if any), optimizer and loss
from data_loader import Dataset, get_dataloader    # grabbing the dataloaders
from run_model import run_model                    # the main training and evaluation loop
from datetime import datetime
# import tensorflow as tf
import re

# # trains and evaluates the neural net by calling run_model.py
def train_and_eval(model, train_loader, valid_loader,
                   learning_rate, epochs,
                   model_outdir, wts, task,
                   metrics_every_iter, restore_chkpt = None, run_suffix = None):
    """
    Contains the powerhouse of the network, ie. the training and validation iterations called through run_model().
    All parameters from the command line/json are parsed and then passed into run_model().
    Performs checkpointing each epoch, saved as 'last.pth.tar' and the best model thus far (based on validation AUC), saved as 'best.pth.tar'

    :param model: (nn.Module)
    :param train_loader: (torch DataLoader)
    :param valid_loader: (torch DataLoader)
    :param learning_rate: (float) - the learning rate, defaults to 1e-05
    :param epochs: (int) - the number of epochs
    :param wts: (tensor) - class weights
    :param model_outdir: (str) - the output directory for checkpointing, checkpoints will be saved as output_dir/task/view/*.tar
    :param restore_chkpt: (str) - the directory to reload the checkpoint, if specified
    :param run_suffix: (str) - suffix to be appended to the event file
    :return:
    """

    # output/task/my_run
    # goes back 3 levels up, putting the name in the same level as the output. two levels up would put them into output/
    recover_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_outdir))) # removes the task and view from the directory
    log_dir = os.path.join(recover_root_dir, "logs")
    run_name = re.split(r'/|\\', model_outdir)[-1]
    # task = re.split(r'/|\\', model_outdir)[-2]
    # have log folder naming structure same as models
    log_fn = os.path.join(log_dir, task, run_name)

    dtnow = datetime.now()
    # dtnow.strftime("%Y%m%d_%H%M%S")
    log_fn = os.path.join(log_fn,  dtnow.strftime("%Y_%m_%d-%H_%M_%S"))

    # make directory if it doesn't exist.
    if not os.path.exists(log_fn):
        os.makedirs(log_fn)
        print('{} does not exist, creating..!'.format(log_fn))
    else:
        print('{} already exists!'.format(log_fn))

    # each tensorboard event file should ideally be saved to a unique folder, else the resulting graph will look like
    # it's time traveling because of overlapping logs
    # if run_suffix:
    #     writer = tf.summary.create_file_writer(log_fn, filename_suffix=run_suffix)
    # else:
    #     writer = tf.summary.create_file_writer(log_fn)

    # use cpu or cuda depending on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_best_val_loss = float('Inf')
    # this needs to be outside of the loop else it'll keep resetting, right? same with the model
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)
    # taken directly from MRNet code
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 5, # how many epochs to wait for before acting
                                                           factor = 0.3, # factor to reduce LR by, LR = factor * LR
                                                           threshold = 1e-4)  # threshold to measure new optimum

    # weight loss by training class positive weights, if use_wts is False then no weights are applied
    # criterion_d = {'bladder': torch.nn.BCEWithLogitsLoss(), 'view': torch.nn.CrossEntropyLoss(), 'granular':torch.nn.CrossEntropyLoss()}
    if wts is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        wts = wts.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight = wts)


    # # TODO: reloading checkpoint
    # if restore_chkpt:
    #     logging.info("Restoring Checkpoint from {}".format(restore_chkpt))
    #     helpers.load_checkpoint(checkpoint = restore_chkpt,
    #                             model = model,
    #                             optimizer = optimizer,
    #                             # scheduler = scheduler,
    #                             epochs = epochs)
    #     # so epochs - loaded_epoch is where we would need to start, right?
    #     logging.info("Starting again at Epoch {}....".format(epochs))
    #     logging.info("Finished Restoring Checkpoint...")

    for epoch in range(epochs):
        logging.info('[Epoch {}]'.format(epoch + 1))
        # main training loop
        epoch_loss, epoch_preds, epoch_labels = run_model(model = model,
                                               loader = train_loader,
                                               optimizer = optimizer,
                                               criterion = criterion,
                                               metrics_every_iter  = metrics_every_iter,
                                               train = True)
        logging.info('[Epoch {}]\t\t Training Average Loss: {:.5f}'.format(epoch + 1, epoch_loss))

        # logging.info('[Epoch {}]\t\tTraining Balanced Accuracy: {:.3f}\t Training Average Loss: {:.5f}'.format(epoch + 1, epoch_auc, epoch_loss))
        # main validation loop
        epoch_val_loss,  epoch_val_preds, epoch_val_labels = run_model(model = model,
                                                             loader = valid_loader,
                                                             optimizer = optimizer,
                                                             criterion = criterion,
                                                             metrics_every_iter = False, # default, just show the epoch validation metrics..
                                                             train = False)

        logging.info('[Epoch {}]\t\t Validation Average Loss: {:.5f}'.format(epoch + 1, epoch_val_loss))

        # logging.info('[Epoch {}]\t\tValidation Balanced Accuracy: {:.3f}\t Validation Average Loss: {:.5f}'.format(epoch + 1, epoch_val_acc, epoch_val_loss))
        scheduler.step(epoch_val_loss) # check per epoch, how does the threshold work?!?!?
        logging.info('[Epoch {}]\t\tOptimizer Learning Rate: {}'.format(epoch + 1, {optimizer.param_groups[0]['lr']}))

        # with writer.as_default():
        #     tf.summary.scalar('Loss/train', epoch_loss, epoch + 1)
        #     tf.summary.scalar('Loss/val', epoch_val_loss, epoch + 1)
            # tf.summary.scalar('BACC/train', epoch_acc, epoch + 1)
            # tf.summary.scalar('BACC/val', epoch_val_acc, epoch + 1)

        # check whether the most recent epoch loss is better than previous best
        # is_best_val_auc = epoch_val_auc >= current_best_val_auc
        is_best_val_loss = epoch_val_loss < current_best_val_loss

        # save state in a dictionary
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 # 'validation_acc': epoch_val_acc,
                 'best_validation_loss': epoch_val_loss,
                 # 'metrics': metrics # read more into this
                 'scheduler_dict': scheduler.state_dict(),
                 'optim_dict': optimizer.state_dict()}

        # save as last epoch
        helpers.save_checkpoint(state,
                                is_best = is_best_val_loss,
                                checkpoint_dir = model_outdir)
                                # epoch = epoch + 1)

        if is_best_val_loss:
        # set new best validation loss
            # current_best_val_auc = epoch_val_auc
            current_best_val_loss = epoch_val_loss
            # logging.info('[Epoch {}]\t\t******New Best Validation:\t AUC: {:.3f}******'.format(epoch + 1, epoch_val_auc))
            logging.info('[Epoch {}]\t\t******New Best Validation Loss: {:.3f}******'.format(epoch + 1, epoch_val_loss))
            helpers.save_checkpoint(state,
                                  is_best = is_best_val_loss,
                                  checkpoint_dir = model_outdir)
                                    # epoch = epoch + 1)
            # torch.save(state, model_outdir)

# the arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type = str, required = True, help = 'root directory')
    parser.add_argument('--manifest_path', type=str, required=True, help = 'absolute path of the manifest file')
    parser.add_argument('--task', type = str, default='granular', help = 'one of granular (6), view (4), or bladder (2)')
    parser.add_argument('--model_out', type=str, required = True, help = 'output directory')
    parser.add_argument('--model', type=str, default='AlexNet', help='one of vgg, resnet, alexnet, squeezenet, desnet, or custom')
    parser.add_argument('--num_epochs', type = int, required = True, help = 'int, the number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 1e-5, help = 'the learning rate')
    parser.add_argument('--metrics_every_iter', type = int, default = False, help = 'calculates metrics every i batches')
    parser.add_argument('--use_wts', dest='use_wts', action='store_true')
    parser.add_argument('--no_wts', dest='use_wts', action='store_false')
    parser.set_defaults(use_wts=True)
    # parser.add_argument('--restore_chkpt', type = str, default = False, help = 'location of checkpoint to reload')
    parser.add_argument('--run_suffix', type=str, default=False, help='suffix to append to end of tensorflow log file')
    parser.add_argument('--batch_size', type=int,default=1, help ='batch size to use in training')
    parser.add_argument('--run_name', type=str, default='my_run', help='name of output directory where checkpoints and log is saved')
    return parser


if __name__ == '__main__':

    # ----- get arguments -----
    args = get_parser().parse_args()

    # ----- reproducibility!!!! -----

    torch.manual_seed(1)

    # ---- some set up -----
    # create model outdir if it doesn't already exist, corresponding to task and view
    model_outdir = os.path.join(args.model_out, args.task, args.run_name)
    # out_directory/{bladder|view|granular}/my_run_name, eg. /Users/delvin/Downloads/sshfs_test/output/bladder/my_run

    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
        print('{} does not exist, creating..!'.format(model_outdir))
    else:
        print('{} already exists!'.format(model_outdir))

    # set up the logger
    helpers.set_logger(os.path.join(model_outdir, 'train.log'))

    # print arguments to log
    logging.info('-' * 20 + 'ARGUMENTS' + '-' * 20)
    for arg in vars(args):
        logging.info(arg + ": " + str(getattr(args, arg)))
    logging.info('-' * 49)

    # check device..
    logging.info('Device: {}'.format(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    # ---- retrieve the training and validation loaders based on the manifest -----
    logging.info('Retrieving Dataloaders..')

    dataloaders = get_dataloader(sets = ['train', 'valid'],
                                 root_dir = args.root_path,
                                 task = args.task,
                                 manifest_path = args.manifest_path,
                                 batch_size= args.batch_size)

    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    logging.info('Finished retrieving dataloaders..')

    # ------ get weights, if specified -----

    if args.use_wts == True:
        logging.info('Calculating Weights for Loss..')
        wts = helpers.get_class_weights(manifest_dir = args.manifest_path, task = args.task)
        logging.info('Weights are {}'.format(wts))
    else:
        logging.info('Not using weights!!!!!!!!')
        wts = None

    # ---- passing the arguments to our workhorse! ------
    logging.info("Training for {} epoch(s)..".format(args.num_epochs))

    train_and_eval(model = Net(task = args.task, mod = args.model),
                   train_loader = train_loader,
                   valid_loader = valid_loader,
                   learning_rate = args.learning_rate,
                   epochs = args.num_epochs,
                   metrics_every_iter = args.metrics_every_iter,
                   wts = wts,
                   task = args.task,
                   model_outdir = model_outdir,
                   run_suffix = args.run_suffix)

    logging.info("Done!!..\n Please check {} for the results!".format(model_outdir))
#


import os
import argparse
import torchvision
import pandas as pd
import torch 
import numpy as np
import time
import pprint
import tqdm
import exp_configs
from src import datasets, models, optimizers, metrics

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def newminimum(exp_id, savedir_base, datadir, name, exp_dict, metrics_flag=True):
    # bookkeeping
    # ---------------

    # get experiment directory
    old_modeldir = os.path.join(savedir_base, exp_id)
    savedir = os.path.join(savedir_base, exp_id, name)

    old_exp_dict = hu.load_json(os.path.join(old_modeldir, 'exp_dict.json'))  

    # TODO: compare exp dict for possible errors:
    # optimizer have to be the same
    # same network, dataset  
    
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)
    
    # set seed
    # ---------------
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # -----------

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = torch.utils.data.DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)


    # Model
    # -----------
    model = models.get_model(exp_dict["model"],
                             train_set=train_set)

    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch)

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    opt_path = os.path.join(savedir, 'opt_state_dict.pth')


    old_model_path = os.path.join(old_modeldir, 'model.pth')
    old_score_list_path = os.path.join(old_modeldir, 'score_list.pkl')
    old_opt_path = os.path.join(old_modeldir, 'opt_state_dict.pth')

    score_list = hu.load_pkl(old_score_list_path)
    model.load_state_dict(torch.load(old_model_path))
    opt.load_state_dict(torch.load(old_opt_path))
    s_epoch = score_list[-1]['epoch'] + 1


    # save current model state for comparison
    minimum = []

    for param in model.parameters():
        minimum.append(param.clone())

    



    # Train & Val
    # ------------
    print('Starting experiment at epoch %d/%d' % (s_epoch, exp_dict['max_epoch']))

    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        # torch.cuda.manual_seed_all(exp_dict['runs']+epoch) not needed since no cuda available

        score_dict = {"epoch": epoch}

        if metrics_flag:
            # 1. Compute train loss over train set
            score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, train_set, metric_name='softmax_loss')
            #                                    metric_name=exp_dict["loss_func"]) 
            # TODO: which loss should be used? (normal or with reguralizer?)

            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set,
                                                        metric_name=exp_dict["acc_func"])

        # 3. Train over train loader
        model.train()
        print("%d - Training model with %s..." % (epoch, exp_dict["loss_func"]))

        s_time = time.time()
        for images,labels in tqdm.tqdm(train_loader):
            # images, labels = images.cuda(), labels.cuda() no cuda available

            opt.zero_grad()
            loss = loss_function(model, images, labels, minimum, 0.1) # just works for custom loss function
            loss.backward()
            opt.step()

        e_time = time.time()

        # Record metrics
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["n_forwards"] = opt.state["n_forwards"]
        score_dict["n_backwards"] = opt.state["n_backwards"]
        score_dict["batch_size"] =  train_loader.batch_size
        score_dict["train_epoch_time"] = e_time - s_time

        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved: %s" % savedir)


        with torch.nograd():
            print('Current distance: %f', metrics.computedistance(minimum, model))
        



    print('Experiment completed')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-ei', '--exp_id', required=True)
    parser.add_argument('-e', '--exp_group_list', nargs='+')

    args = parser.parse_args()

    # Collect experiments
    
    # select exp group
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]
    


    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        newminimum(exp_id=args.exp_id,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                name=args.name,
                exp_dict=exp_dict)

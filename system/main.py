#!/usr/bin/env python
import os
import torch
import argparse
import time
import warnings
import numpy as np
import logging
import timm
import random
from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverkd import FedKD
from flcore.servers.servertgp import FedTGP
from flcore.servers.serversce import Fedsce

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def set_random_seed(seed, deterministic=True):
    global current_seed
    current_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(0)

def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.models
      
        if args.model_family == "Cifar10":    
            args.models = [
                "timm.create_model('resnet18', pretrained=False)",
                "timm.create_model('resnet34', pretrained=False)",
                "timm.create_model('resnet50', pretrained=False)",
                "timm.create_model('vit_base_patch16_224', img_size=32, pretrained=False)",
                "timm.create_model('vit_base_patch32_224', img_size=32, pretrained=False)"
            ] 

        elif args.model_family == "Cifar100":    
            args.models = [
                "timm.create_model('resnet18', pretrained=False)",
                "timm.create_model('resnet34', pretrained=False)",
                "timm.create_model('resnet50', pretrained=False)",
                "timm.create_model('vit_base_patch16_224', img_size=32, pretrained=False)",
                "timm.create_model('vit_base_patch32_224', img_size=32, pretrained=False)"
            ] 

        elif args.model_family == "Digit5_Pretrained":    
            args.models = [
                "timm.create_model('resnet18', pretrained=True)",
                "timm.create_model('resnet34', pretrained=True)",
                "timm.create_model('resnet50', pretrained=True)",
                "timm.create_model('vit_base_patch16_224', img_size=32, pretrained=True)",
                "timm.create_model('vit_base_patch32_224', img_size=32, pretrained=True)"
            ]  

        elif args.model_family == "Digit5":    
            args.models = [
                "timm.create_model('resnet18', pretrained=False)",
                "timm.create_model('resnet34', pretrained=False)",
                "timm.create_model('resnet50', pretrained=False)",
                "timm.create_model('vit_base_patch16_224', img_size=32, pretrained=False)",
                "timm.create_model('vit_base_patch32_224', img_size=32, pretrained=False)"
            ] 

        elif args.model_family == "Office_10":  
            args.models = [
                "timm.create_model('resnet18', pretrained=False)",   
                "timm.create_model('resnet34', pretrained=False)",  
                "timm.create_model('resnet50', pretrained=False)",   
                "timm.create_model('vit_base_patch16_224', img_size=64, pretrained=False)", 
                "timm.create_model('vit_base_patch32_224', img_size=64, pretrained=False)"
            ] 

        elif args.model_family == "DomainNet":  
            args.models = [
                "timm.create_model('resnet18', pretrained=False)",
                "timm.create_model('resnet34', pretrained=False)",
                "timm.create_model('resnet50', pretrained=False)",
                "timm.create_model('vit_base_patch16_224', img_size=64, pretrained=False)",
                "timm.create_model('vit_base_patch32_224', img_size=64, pretrained=False)"
            ] 

        else:
            raise NotImplementedError
            
        for model in args.models:
            print(model)

        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)
            
        elif args.algorithm == "Fedsce":
            server = Fedsce(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!") 
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="9")
    parser.add_argument('-data', "--dataset", type=str, default="Digit5")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="Digit5")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedKTL-stable-diffusion")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False) 
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedProto & FedSCE
    parser.add_argument('-lam', "--lamda", type=float, default=0.1)

    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.95)
    # FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)


    parser.add_argument('-gnu', "--gen_num", type=int, default=2,
                        help="Each label corresponds to the number of generated images")
    parser.add_argument('-lam2', "--lamda2", type=float, default=0.1)
    parser.add_argument('-cw', "--cw", type=str, default="avg",help="Client prototype aggregation")
    parser.add_argument('-norm', "--norm", type=float, default=0,help="regularization coefficient")
    parser.add_argument('-m_se', "--m_se",type=int, default=1,help="Whether to use MSE regular")
    parser.add_argument('-s_g_p', "--s_g_p", type=str, default="s_g_p",help="Whether to use Server global protos")
    parser.add_argument('-corr', "--corr", type=int, default=1,help="Whether to use correct prototype")
    # FedSCE
    parser.add_argument('-rank','--rank', type=int, default=3, help="rank parameter")
    parser.add_argument('-l_i', "--layer_idx", type=int, default=30,help="layer number of subspace")
    parser.add_argument('-gap',"--gap", type=int, default=5, help="The gap between subspace update")
    
    
    parser.add_argument('-sf',"--select_frac", type=float, default=0.5, help="select_frac")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"


    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_family))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    print("=" * 50)




    run(args)



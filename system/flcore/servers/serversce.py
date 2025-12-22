import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientsce import clientsce
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
from flcore.trainmodel.models import BaseHeadSplit, BaseHeadSplit_global
import tqdm
from torch.utils.data import DataLoader, TensorDataset

class Fedsce(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientsce)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = BaseHeadSplit(args, 0).to(args.device)            #第一个模型作为全局模型  
            # global_model = BaseHeadSplit_global(args, 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)').to(args.device)            #第一个模型作为全局模型  
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
            total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
            print(f"Total global_model Parameters: {total_params}")
        print("Finished creating server and clients.")

        self.Budget = []


    def train(self):
        for i in tqdm.tqdm(range(self.global_rounds+1)):
            s_t = time.time()
            self.selected_clients = self.select_clients()
    
            for client in self.selected_clients:
                client.train(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()   

    #------------------------------------------------------aggregate------------------------------------------------------            
            self.receive_ids() 
            self.W_aggregate_parameters() 
    #------------------------------------------------------aggregate------------------------------------------------------            

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nBest averaged clients accurancy.")
        print(max(self.rs_avg_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    #------------------------------------------------------evaluate------------------------------------------------------  
    def evaluate(self, acc=None, loss=None):
            stats = self.test_metrics()  
            stats_train = self.train_metrics()
            stats_train_acc = self.train_acc()

            test_acc = sum(stats[2])*1.0 / sum(stats[1])
            test_auc = sum(stats[3])*1.0 / sum(stats[1])
            train_acc = sum(stats_train_acc[2])*1.0 / sum(stats_train_acc[1])
            train_auc = sum(stats_train_acc[3])*1.0 / sum(stats_train_acc[1])
            train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
            accs = [a / n for a, n in zip(stats[2], stats[1])]
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
            avg_acc = sum(accs) / len(accs)
            train_accs = [a / n for a, n in zip(stats_train_acc[2], stats_train_acc[1])]
            train_aucs = [a / n for a, n in zip(stats_train_acc[3], stats_train_acc[1])]
            train_avg_acc = sum( train_accs) / len( train_accs)
            
            if acc == None:
                self.rs_test_acc.append(test_acc)
                self.rs_avg_acc.append(avg_acc)  
            else:
                acc.append(test_acc)
            
            if loss == None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)

            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Clients Test Accurancy: {:.4f}".format(avg_acc))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Averaged Clients Train Accurancy: {:.4f}".format(train_avg_acc))
            print("Averaged Train Accurancy: {:.4f}".format(train_acc))
            print("Averaged Train AUC: {:.4f}".format(train_auc))
            # self.print_(test_acc, train_acc, train_loss)
            print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()  
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc  

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    
    def train_acc(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.train_acc()  
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Train_Acc: {ct*1.0/ns}, Train_AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc  
    #------------------------------------------------------evaluate------------------------------------------------------ 


    #------------------------------------------------------aggregate_parameters------------------------------------------------------

    def W_aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()

        F1_sum = 0
        F2_sum = 0
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            F1_sum += client.F1
            F2_sum += client.F2

        W_list = []
        W_sum = 0
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            W1 = client.F1 / F1_sum
            W2 = client.F2 / F2_sum
            W = W1 + W2
            W_list.append(W)
            W_sum += W

        for idx, cid in enumerate(self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'global_model', client.save_folder_name)
            weight = W_list[idx] / W_sum
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * weight

        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    #------------------------------------------------------aggregate_parameters------------------------------------------------------





import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import json
from torch.autograd import Variable
import random

class clientsce(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)  
        torch.manual_seed(0)

        self.lamda = args.lamda
        self.layer_idx=args.layer_idx
        self.rank=args.rank
        self.gap=args.gap
        self.KL = nn.KLDivLoss()

        model = load_item(self.role, 'model', self.save_folder_name) 
        projection=self.initialize_projection_mat(model)    
        save_item(projection, self.role, 'projection', self.save_folder_name) 

    def train(self, global_round):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        global_model_before = copy.deepcopy(global_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        for param in model.parameters():
            param.requires_grad = True

        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        prev_params = [p.clone() for p in model.parameters()]  
        projection=load_item(self.role,'projection', self.save_folder_name)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)  #torch.Size([64, 512])
                output = model.head(rep)
                CE_loss = self.loss(output, y)
                output_g = global_model(x)
                CE_loss_g = self.loss(output_g, y)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / (CE_loss + CE_loss_g)   
                loss = CE_loss + L_d
                loss_g = CE_loss_g + L_d_g
                reg_loss=self.projection_mat_loss(model, prev_params, projection)
                loss+=reg_loss * self.lamda
                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()

        self.calculate_F1_F2(global_model_before, global_model, trainloader)

        if global_round % self.gap == 0:
            projection=self.update_projection(model,prev_params,projection)
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        save_item(projection, self.role, 'projection', self.save_folder_name) 
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
    def test_metrics(self):   
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)    
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    def train_acc(self):   
        trainloaderfull = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_acc = 0
        train_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in trainloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return train_acc, train_num, auc


    def initialize_projection_mat(self, model):


        params = list(model.parameters())
        total_dim = 0
        
        for param in params[-self.layer_idx:]:
            total_dim += param.numel()

        random_matrix = torch.randn(total_dim, self.rank, device=self.device)
        projection = random_matrix.to(self.device)
        
        return projection

    def update_projection(self,model,prev_params,projection):
        delta_w_combined = []
        
        model_params = list(model.parameters())

        for param, prev_param in zip(model_params[-self.layer_idx:], prev_params[-self.layer_idx:]):
            if param.requires_grad: 
                delta_w = (param - prev_param).reshape(-1)  
                delta_w_combined.append(delta_w)

        delta_w_combined = torch.cat(delta_w_combined, dim=0)  
        combined_matrix = torch.cat((projection, delta_w_combined.unsqueeze(1)), dim=1) 

        U, _, _ = torch.linalg.svd(combined_matrix, full_matrices=False)

        
        projection = U[:, :self.rank]  

        return projection

    def projection_mat_loss(self, model, prev_params, projection):

        delta_w_combined = []
        model_params = list(model.parameters())

        for param, prev_param in zip(model_params[-self.layer_idx:], prev_params[-self.layer_idx:]):  
            if param.requires_grad: 
                delta_w = (param - prev_param).reshape(-1) 
                delta_w_combined.append(delta_w)

        delta_w_combined = torch.cat(delta_w_combined, dim=0) 
        
        delta_w_proj = projection @ (projection.T @ delta_w_combined)

        reg_loss = torch.norm(delta_w_combined - delta_w_proj) ** 2
        
        return reg_loss
    
    def calculate_F1_F2(self, global_model_before, global_model_after, trainloader):

        param_differences = [(p_after - p_before).norm(2) for p_before, p_after in zip(global_model_before.parameters(), global_model_after.parameters())]
        self.F1 = torch.sqrt(sum([diff ** 2 for diff in param_differences]))

        global_model_before.eval()
        global_model_after.eval()
        feature_differences = []
        with torch.no_grad():
            for x, _ in trainloader:
                x = x.to(self.device)

                feature_before = global_model_before.base(x)
                feature_after = global_model_after.base(x)

                feature_differences.append((feature_after - feature_before).norm(2))

        self.F2 = torch.sqrt(sum([diff ** 2 for diff in feature_differences]))
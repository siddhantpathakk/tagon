from models.model import GNN_SR_Net
from models.GCN import GCN
import torch
import numpy as np
import csv
import random
from utils.metric import *
import pickle
import time
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,config,node_num,relation_num,u2v,u2vc,v2u,v2vc,device):
        self.u2v,self.u2vc,self.v2u,self.v2vc = u2v,u2vc,v2u,v2vc
        v_list_ = list(self.u2v.values())
        v_list_ = sorted(set([v_list_[i][j] for i in range(len(v_list_)) for j in range(len(v_list_[i]))]))
        self.item_indexes = torch.tensor(v_list_).to(device)
        item_num = len(v_list_)
        self.item_set = set(self.v2u.keys())
        self.gnn_sr_model = GNN_SR_Net(config,item_num,node_num,relation_num,gcn=GCN,device=device)
        self.device = device
        self.arg = config
        self.node_num = node_num
        self.optimizer = torch.optim.Adam(self.gnn_sr_model.parameters(), 
                                          lr=config.learning_rate, 
                                          weight_decay=config.l2,
                                          )
        self.short_term_window_num = 3
 
    def Negative_Sampling(self,user2item,batch_users,item_set):
        negatives_np = list()
        for i in range(batch_users.shape[0]):
            user_item_set = set(user2item[batch_users[i]])
            difference_set = item_set - user_item_set
            negtive_list_ = random.sample(sorted(difference_set),self.arg.H)
            negatives_np.append(negtive_list_)
        negatives_np = np.array(negatives_np)
        return negatives_np

    def Extract_SUBGraph(self, user_no, seq_no, sub_seq_no=None, node2ids=None, hop=2, test=False):
        '''
         vc (O)     vc(X)       vc(O)    
        u{v1,.. => {u1,... => {v5,v6,... 
        _______    _______    _______ ....
         0-hop      1-hop      2-hop

         edge type: uv 0 ; vu 1 ; vvc 2 ;  vcv 3
        '''
        if sub_seq_no is not None:
            origin_seq_no = seq_no
            seq_no = sub_seq_no

        if node2ids is  None:
            node2ids = dict()

        edge_index,edge_type = list(),list()
        update_set,index,memory_set = list(),0,list()
        for i in range(hop):
            if i == 0:
                #uv
                for j in range(user_no.shape[0]):
                    if user_no[j] not in node2ids:
                        node2ids[user_no[j]] = index
                        index +=1
                    user_node_ids = node2ids[user_no[j]]
                    for k in range(seq_no[j,:].shape[0]):
                        if seq_no[j,k] not in node2ids:
                            node2ids[seq_no[j,k]] = index
                            index +=1
                        item_node_ids = node2ids[seq_no[j,k]]
                        edge_index.append([user_node_ids,item_node_ids])
                        edge_type.append(0)
                        edge_index.append([item_node_ids,user_node_ids])
                        edge_type.append(1)
                        #vc
                        vc = self.v2vc[seq_no[j,k]]
                        if vc not in node2ids:
                            node2ids[vc] = index
                            index +=1      
                        vc_node_ids = node2ids[vc] 
                        edge_index.append([item_node_ids,vc_node_ids])
                        edge_type.append(2)
                        edge_index.append([vc_node_ids,item_node_ids])
                        edge_type.append(3)       
                        # update       
                        update_set.append(seq_no[j,k])
                        #memory
                        memory_set.append(user_no[j])

                update_set = list(set(update_set)) #v
                memory_set = set(memory_set) #u

                if sub_seq_no is not None:
                    for j in range(user_no.shape[0]):
                        user_node_ids = node2ids[user_no[j]]
                        for k in range(origin_seq_no[j,:].shape[0]):
                            if origin_seq_no[j,k] not in node2ids:
                                node2ids[origin_seq_no[j,k]] = index
                                index +=1
                            if origin_seq_no[j,k] not in set(update_set):
                                item_node_ids = node2ids[origin_seq_no[j,k]]
                                edge_index.append([user_node_ids,item_node_ids])
                                edge_type.append(0)
                                edge_index.append([item_node_ids,user_node_ids])
                                edge_type.append(1)
                                #vc
                                vc = self.v2vc[origin_seq_no[j,k]]
                                if vc not in node2ids:
                                    node2ids[vc] = index
                                    index +=1      
                                vc_node_ids = node2ids[vc] 
                                edge_index.append([item_node_ids,vc_node_ids])
                                edge_type.append(2)
                                edge_index.append([vc_node_ids,item_node_ids])
                                edge_type.append(3)      

            elif i != 0 and i % 2 != 0:
                # vu
                new_update_set,new_memory_set = list(),list()
                for j in range(len(update_set)):
                    item_node_ids = node2ids[update_set[j]]
                    u_list = self.v2u[update_set[j]]
                    for k in range(len(u_list)):
                        if u_list[k] not in node2ids:
                            node2ids[u_list[k]] = index
                            index +=1
                        if u_list[k] not in memory_set:
                            user_node_ids = node2ids[u_list[k]]
                            edge_index.append([item_node_ids,user_node_ids])
                            edge_type.append(1)                       
                            edge_index.append([user_node_ids,item_node_ids])
                            edge_type.append(0)            
                            new_update_set.append(u_list[k])
                memory_set = set(update_set) #v
                update_set = new_update_set #u
            elif i != 0 and i % 2 == 0:
                #uv      
                for j in range(len(update_set)):
                    user_node_ids = node2ids[update_set[j]]
                    v_list = self.u2v[update_set[j]]
                    for k in range(len(v_list)):
                        if v_list[k] not in node2ids:
                            node2ids[v_list[k]] = index
                            index +=1
                        if v_list[k] not in memory_set:
                            item_node_ids = node2ids[v_list[k]]
                            edge_index.append([item_node_ids,user_node_ids])
                            edge_type.append(1)                       
                            edge_index.append([user_node_ids,item_node_ids])
                            edge_type.append(0)            
                            new_update_set.append(v_list[k])
                memory_set = set(update_set)  #u
                update_set = new_update_set   #v
        
        edge_index = torch.t(torch.tensor(edge_index).to(self.device))
        edge_type = torch.tensor(edge_type).to(self.device)
        node_no = torch.tensor(sorted(list(node2ids.values()))).to(self.device)
        
        # new_user_no,new_seq_no
        new_user_no,new_seq_no = list(),list()
        for i in range(user_no.shape[0]):
            new_user_no.append(node2ids[user_no[i]])
        for i in range(seq_no.shape[0]):
            new_seq_no.append([node2ids[seq_no[i,j]] for j in range(seq_no[i,:].shape[0])])
        new_user_no,new_seq_no = np.array(new_user_no),np.array(new_seq_no)

        return new_user_no,new_seq_no,edge_index,edge_type,node_no,node2ids
            
    def _Eval_Draw_Graph_(self,batch_users,test_set,rating_pred,edge_index,edge_type,attn_weight_list,node2ids,topk=10):
        batch_users = batch_users.cpu().data.numpy().copy()
        rating_pred = rating_pred.cpu().data.numpy().copy()
        edge_index = edge_index.cpu().data.numpy().copy()
        edge_type = edge_type.cpu().data.numpy().copy()
        attn_weight_list = [attn_weight_list[i].cpu().data.numpy().copy() for i in range(len(attn_weight_list))]

        # build uv2attn
        uv2attn,attn_list = dict(),list()
        for i in range(edge_index.shape[1]):
            if edge_type[i] == 0:
                user,item = edge_index[0,i],edge_index[1,i]
                uv_tocken = str(user) + '-' + str(item)
                if uv_tocken not in uv2attn:
                    uv2attn[uv_tocken] = [attn_weight_list[j][i,0] for j in range(len(attn_weight_list))]
                    for j in range(len(attn_weight_list)):
                        attn_list.append(attn_weight_list[j][i,0])
        mean_attn = sorted(attn_list)[int(len(attn_list)*0.5)]

        # Rating Graph
        user = batch_users[50]
        top10_list = test_set[50][:100]
        ratings = list(rating_pred[50,top10_list])

        # Attention Graph
        attns = list()
        for i in range(len(top10_list)):
            if top10_list[i] not in node2ids:
                attn_value = [mean_attn for j in range(len(attn_weight_list))]
                # print(000)
            else:                
                item = node2ids[top10_list[i]]
                uv_tocken = str(user) + '-' + str(item)
                if uv_tocken not in uv2attn:
                    attn_value = [mean_attn for j in range(len(attn_weight_list))]
                    # print(333)
                else:
                    attn_value = uv2attn[uv_tocken]
                    # print(111)
            attns.append(attn_value)               

        item_list,rating_list,attn_list = list(),list(),list()
        attn_list = [list() for i in range(len(attn_weight_list))]
        for i in range(len(top10_list)):
            if top10_list[i] not in item_list:
                item_list.append(top10_list[i])
                rating_list.append(ratings[i])
                for j in range(len(attn_weight_list)):
                    attn_list[j].append(attns[i][j])

    def Evaluation(self, users_np_test, sequences_np_test, test_set):
        
        #short term part
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i+short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        batch_num = int(users_np_test.shape[0]/self.arg.batch_size)+1
        data_index = np.arange(users_np_test.shape[0])
        self.pred_list = None
        for batch_ in range(batch_num):
            start_index , end_index = batch_*self.arg.batch_size , (batch_+1)*self.arg.batch_size
            batch_record_index = data_index[start_index : end_index]

            batch_users = users_np_test[batch_record_index]
            batch_sequences = sequences_np_test[batch_record_index]
 
            #Extracting SUBGraph (long term)
            batch_users,batch_sequences,edge_index,edge_type,node_no,node2ids = self.Extract_SUBGraph(batch_users,batch_sequences,sub_seq_no=None)

            #Extracting SUBGraph (short term)
            short_term_part = []
            for i in range(len(short_term_window)):
                if i != len(short_term_window)-1:
                    sub_seq_no = batch_sequences[:,short_term_window[i]:short_term_window[i+1]]
                    _,_,edge_index,edge_type,_,_ = self.Extract_SUBGraph(batch_users,batch_sequences,sub_seq_no=sub_seq_no,node2ids=node2ids)
                    short_term_part.append((edge_index,edge_type))

            batch_users = torch.tensor(batch_users).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

            X_user_item = [batch_users,batch_sequences,self.item_indexes]
            X_graph_base = [edge_index,edge_type,node_no,short_term_part]

            rating_pred = self.gnn_sr_model(X_user_item,X_graph_base,for_pred=True)

            attn_weight_list = self.gnn_sr_model.attn_weight_list
            self._Eval_Draw_Graph_(batch_users,test_set,rating_pred,edge_index,edge_type,attn_weight_list,node2ids,topk=10)

            rating_pred = rating_pred.cpu().data.numpy().copy()

            ind = np.argpartition(rating_pred, -self.arg.topk)
            ind = ind[:, -self.arg.topk:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            if batch_ == 0:
                self.pred_list = batch_pred_list
            else:
                self.pred_list = np.append(self.pred_list, batch_pred_list, axis=0)

        precision, recall, MAP, ndcg, hr = [], [], [], [], []
        for k in [10, 20]:
            precision.append(precision_at_k(test_set, self.pred_list, k))
            recall.append(recall_at_k(test_set, self.pred_list, k))
            MAP.append(mapk(test_set, self.pred_list, k))
            ndcg.append(ndcg_k(test_set, self.pred_list, k))
            hr.append(hit_ratio_at_k(test_set, self.pred_list, k))
        return precision, recall, MAP, ndcg, hr

    def Eval_Draw_Graph_(self,users_np_test,sequences_np_test,test_set,uid2locid_time,threshold_rate=0.8,restrict_user_num=100):
        uid2locid = dict()
        for i in range(users_np_test.shape[0]):
            locid_time_ = uid2locid_time[users_np_test[i]]
            new_locid_time_ = [locid_time_[j][0] for j in range(len(locid_time_))]
            uid2locid[users_np_test[i]] = new_locid_time_
        for k in range(users_np_test.shape[0]):
            user_ = users_np_test[k]
            seq_ = uid2locid[user_]
            user_i_canidate2seq = dict()
            for i in range(users_np_test.shape[0]):
                user_i = users_np_test[i]
                seq_i = uid2locid[user_i]
                if user_i != user_: 
                    intersection_ = set(seq_) & set(seq_i)
                    union_ = set(seq_) 
                    cover_rate_ = len(intersection_)/len(union_)
                    if cover_rate_ >= threshold_rate:
                        user_i_canidate2seq[user_i] = (list(sequences_np_test[i,:]),test_set[i])
            user_i_canidate_list = list(user_i_canidate2seq.keys())
            if len(user_i_canidate_list) >= restrict_user_num:
                select_user_num = restrict_user_num
                select_user_i = random.sample(user_i_canidate_list,select_user_num)
                new_batch_users,new_batch_sequences = list(),list()
                new_batch_users.append(user_)
                new_batch_sequences.append(list(sequences_np_test[i,:]))
                for j in range(len(select_user_i)):
                    seqX,seqY = user_i_canidate2seq[select_user_i[j]]
                    new_batch_users.append(select_user_i[j])
                    new_batch_sequences.append(seqX)
                new_batch_users = np.array(new_batch_users)
                new_batch_sequences = np.array(new_batch_sequences)    
                
                #Extracting SUBGraph
                batch_users,batch_sequences,edge_index,edge_type,node_no = self.Extract_SUBGraph(new_batch_users,new_batch_sequences)
                batch_users = torch.tensor(batch_users).to(self.device)
                batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

                X_user_item = [batch_users,batch_sequences,self.item_indexes]
                X_graph_base = [edge_index,edge_type,node_no]

                rating_pred = self.gnn_sr_model(X_user_item,X_graph_base,for_pred=True)
                attn_weight_list = self.gnn_sr_model.attn_weight_list

                edge_index = edge_index.cpu().data.numpy().copy()
                edge_type = edge_type.cpu().data.numpy().copy()
                for j in range(len(attn_weight_list)):
                    attn_weight_list[j] = attn_weight_list[j].cpu().data.numpy().copy()
                edge_index,edge_type,attn_weight_list = self._Eval_Draw_Graph_Filter_(edge_index,edge_type,attn_weight_list)

                for j in range(len(attn_weight_list)):
                    user,seq_item,seq_vc,attn_value_1,attn_value_2 = list(),list(),list(),list(),list()
                    for h in range(edge_type.shape[0]):
                        type_ = edge_type[h]
                        attn_weight_ = attn_weight_list[j][h]
                        if type_ == 0:
                            user.append(edge_index[0,h])
                            seq_item.append(edge_index[1,h])
                            attn_value_1.append(attn_weight_)
                        # elif type_ == 4:
                        #     seq_item.append(edge_index[0,i])
                        #     seq_vc.append(edge_index[1,i])
                        #     attn_value_2.append(attn_weight_)

    def _Eval_Draw_Graph_Filter_(self,edge_index,edge_type,attn_weight_list):
        v2u_list_,user_set_ = dict(),list()
        for i in range(edge_index.shape[1]):
            node1_,node2_ = edge_index[0,i],edge_index[1,i]
            edge_type_ = edge_type[i]
            if edge_type_ == 0:
                if node2_ not in v2u_list_:
                    v2u_list_[node2_] = list()
                else:
                    v2u_list_[node2_].append(node1_)
                user_set_.append(node1_)
        user_set_ = set(user_set_)
        cover_rate_list_ = list()
        new_edge_index,new_edge_type,new_attn_weight_list = list(),list(),[list(),list()]
        for i in range(edge_index.shape[1]):
            node1_,node2_ = edge_index[0,i],edge_index[1,i]
            edge_type_ = edge_type[i]
            if edge_type_ == 0:
                intersection_ = set(v2u_list_[node2_]) & user_set_
                union_ = user_set_
                cover_rate_ = len(intersection_)/len(union_)
                cover_rate_list_.append(cover_rate_)
        filter_rate = sorted(cover_rate_list_)[:int(len(cover_rate_list_)*0.8)][-1]
        for i in range(edge_index.shape[1]):
            node1_,node2_ = edge_index[0,i],edge_index[1,i]
            edge_type_ = edge_type[i]
            if edge_type_ == 0:
                intersection_ = set(v2u_list_[node2_]) & user_set_
                union_ = user_set_
                cover_rate_ = len(intersection_)/len(union_)
                if cover_rate_ >= filter_rate:
                    new_edge_index.append([node1_,node2_])
                    new_edge_type.append(edge_type_)
                    new_attn_weight_list[0].append(attn_weight_list[0][i][0])
                    new_attn_weight_list[1].append(attn_weight_list[1][i][0])
        new_edge_index = np.array(new_edge_index).T
        new_edge_type = np.array(new_edge_type)
        return new_edge_index,new_edge_type,new_attn_weight_list

    def Eval_New_User_Insert(self,train_part,test_part,choosing_rate=0.7,save=True):
        '''
        add new user in training
        '''
        users_np,sequences_np_train = train_part[0],train_part[1]
        sequences_np_test,test_set,uid_list_ = test_part[0],test_part[1],test_part[2]
        users_np_test = np.array(uid_list_)
        user_set = set(list(users_np))

        new_user_set = set(random.sample(list(user_set),int(len(user_set)*choosing_rate)))
        old_user_set = user_set - new_user_set

        new_user_np,new_sequences_np_train = list(),list()
        new_sequences_np_test,new_test_set,new_uid_list_ = list(),list(),list()
        for i in range(users_np.shape[0]):
            if users_np[i] in old_user_set:
                new_user_np.append(users_np[i])
                new_sequences_np_train.append(sequences_np_train[i,:])
        
        for i in range(users_np_test.shape[0]):
            if users_np_test[i] in new_user_set:
                new_uid_list_.append(users_np_test[i])
                new_sequences_np_test.append(sequences_np_test[i,:])
                new_test_set.append(test_set[i])

        new_user_np = np.array(new_user_np)
        new_sequences_np_train = np.array(new_sequences_np_train)
        new_sequences_np_test = np.array(new_sequences_np_test)

        if save:
            ig_eval_new_user = {
                'new_user_np':new_user_np,
                'new_sequences_np_train':new_sequences_np_train,
                'new_sequences_np_test':new_sequences_np_test,
                'new_test_set':new_test_set,
                'new_uid_list_':new_uid_list_,
            }
            file = open(self.arg.out_path + 'ig_eval_new_user.pickle', 'wb')
            pickle.dump(ig_eval_new_user, file)
            file.close()   

        new_user_np = ig_eval_new_user['new_user_np']
        new_sequences_np_train = ig_eval_new_user['new_sequences_np_train']
        new_sequences_np_test = ig_eval_new_user['new_sequences_np_test']
        new_test_set = ig_eval_new_user['new_test_set']
        new_uid_list_ = ig_eval_new_user['new_uid_list_']

        return new_user_np,new_sequences_np_train,new_sequences_np_test,new_test_set,new_uid_list_

    def Eval_TSNE(self,user_emd_batch_list,item_emd_batch_list):
        user_emd_list,item_emd_list = list(),list()
        for i in range(len(user_emd_batch_list)):
            bz_user_emd = user_emd_batch_list[i].cpu().data.numpy().copy()
            bz_item_emd = item_emd_batch_list[i].cpu().data.numpy().copy()
            
            for j in range(bz_user_emd.shape[0]):
                user_emd,seq_item_emd = bz_user_emd[j,:],bz_item_emd[j,:,:]
                user_emd_list.append(list(user_emd))
                # print(list(user_emd))
                
                for k in range(seq_item_emd.shape[0]):
                    item_emd = seq_item_emd[k,:]
                    item_emd_list.append(list(item_emd))
                    # print(list(item_emd))
                    
        shadow_list,user_index,item_index = list(), 0, 0
        
        with open('tSNE_data1.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for i in range(len(user_emd_list)):
                shadow_list.append(user_index)
                user_index +=1
                tsv_writer.writerow(user_emd_list[i])
            for i in range(len(item_emd_list)):
                shadow_list.append(item_index)
                item_index +=1
                tsv_writer.writerow(item_emd_list[i])    
                            
        with open(self.arg.out_path + 'tSNE_data2.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['node type'])
            for i in range(len(shadow_list)):
                tsv_writer.writerow([shadow_list[i]])

    def save_model(self, path_name):
        torch.save(self.gnn_sr_model.state_dict(), path_name)

    def train(self,train_part,test_part):
        
        loss_list = list()
        precision_list,recall_list,MAP_list,ndcg_list,hr_list = list(),list(),list(),list(),list()
        precision_list_20,recall_list_20,MAP_list_20,ndcg_list_20,hr_list_20 = list(),list(),list(),list(),list()
        
        users_np,sequences_np_train,sequences_np_test,test_set,uid_list_ = self.Eval_New_User_Insert(train_part,test_part,choosing_rate=0.7,save=True)
        # users_np,sequences_np_train = train_part[0],train_part[1]
        # sequences_np_test,test_set,uid_list_ = test_part[0],test_part[1],test_part[2]
        uid2locid_time = test_part[-1]
        users_np_test = np.array(uid_list_)
        
        train_num = users_np.shape[0]
        record_indexes = np.arange(train_num)

        total_loss, epoch_loss = 0.0, 0.0        
        batch_num = int(train_num/self.arg.batch_size) + 1 

        #short term part
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i + short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]

        print('Epoch\tTime\tLoss\t\tP@10\tR@10\tMAP@10\tNDCG@10\tHR@10\t\tP@20\tR@20\tMAP@20\tNDCG@20\tHR@20')

        for epoch_ in range(self.arg.epoch_num):
            
            start = time.time()
            
            self.gnn_sr_model.train()

            user_emd_batch_list,item_emd_batch_list = list(), list()
            
            for batch_ in range(batch_num):
                start_index, end_index = batch_ * self.arg.batch_size, (batch_+1) * self.arg.batch_size
                batch_record_index = record_indexes[start_index : end_index]

                batch_users = users_np[batch_record_index]
                batch_neg = self.Negative_Sampling(self.u2v,batch_users,self.item_set)
                 
                batch_sequences_train = sequences_np_train[batch_record_index]
                batch_sequences,batch_targets = batch_sequences_train[:,:self.arg.L] , batch_sequences_train[:,self.arg.L:]

                # Extracting SUBGraph (long term)
                batch_users,batch_sequences,edge_index,edge_type,node_no,node2ids = self.Extract_SUBGraph(batch_users,batch_sequences,sub_seq_no=None)
                
                # Extracting SUBGraph (short term)
                short_term_part = []
                for i in range(len(short_term_window)):
                    if i != len(short_term_window)-1:
                        sub_seq_no = batch_sequences[:,short_term_window[i]:short_term_window[i+1]]
                        _,_,edge_index,edge_type,_,_ = self.Extract_SUBGraph(batch_users,batch_sequences,sub_seq_no=sub_seq_no,node2ids=node2ids)
                        short_term_part.append((edge_index,edge_type))
                   
                batch_users = torch.tensor(batch_users).to(self.device)
                batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)
                batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(self.device)
                batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(self.device)

                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

                X_user_item = [batch_users,batch_sequences,items_to_predict]
                X_graph_base = [edge_index,edge_type,node_no,short_term_part]

                pred_score,user_emb,item_embs_conv = self.gnn_sr_model(X_user_item,X_graph_base,for_pred=False)
                # print(pred_score.shape)
                # print('pred_score',pred_score)
                user_emd_batch_list.append(user_emb)
                item_emd_batch_list.append(item_embs_conv)

                (targets_pred, negatives_pred) = torch.split(pred_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

                # RAGCN loss (long term)
                gcn_loss = 0
                for gconv in self.gnn_sr_model.conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    gcn_loss += torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                gcn_loss = gcn_loss/len(self.gnn_sr_model.conv_modulelist)

                # RAGCN loss (short term)
                short_gcn_loss = 0
                for gconv in self.gnn_sr_model.short_conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    short_gcn_loss += torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                short_gcn_loss = short_gcn_loss/len(self.gnn_sr_model.short_conv_modulelist)            

                # Combine GCN short term and long term loss
                gcn_loss += short_gcn_loss
                
                # BPR loss
                loss = -torch.log(torch.sigmoid(targets_pred - negatives_pred) + 1e-8)
                loss = torch.mean(torch.sum(loss))
                
                # Total loss = BPR loss + RAGCN loss
                loss = loss + (900 * gcn_loss)
                
                if self.arg.block_backprop:
                    loss = loss * 0 # needed in case to block backpropagation
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # TODO: check whether should division be done here or not
                epoch_loss /= batch_num
                
                total_loss = epoch_loss
            
            
            self.epoch_ = epoch_
            self.Eval_Draw_Graph_(users_np_test,sequences_np_test,test_set,uid2locid_time)
            self.Eval_TSNE(user_emd_batch_list,item_emd_batch_list)
            
            time_ = time.time()
            _, rem = divmod(time_-start, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = "{:0>2}m{:0>2}s".format(int(minutes),int(seconds))
            
            loss_list.append(total_loss)
            
            if (epoch_ +1) % 1 == 0:
                self.gnn_sr_model.eval()
                precision, recall, MAP, ndcg, hr = self.Evaluation(users_np_test,sequences_np_test,test_set)      
                
                precision_list.append(precision[0])
                recall_list.append(recall[0])
                MAP_list.append(MAP[0])
                ndcg_list.append(ndcg[0])
                hr_list.append(hr[0])
                
                precision_list_20.append(precision[1])
                recall_list_20.append(recall[1])
                MAP_list_20.append(MAP[1])
                ndcg_list_20.append(ndcg[1])
                hr_list_20.append(hr[1])
                        
                # TODO: check whether should division be done here or not with batch_num for loss value
                print(f'{epoch_+1}\t{time_str}\t{total_loss/batch_num:.3f}\t\t{precision[0]:.3f}\t{recall[0]:.3f}\t{MAP[0]:.3f}\t{ndcg[0]:.3f}\t{hr[0]:.3f}\t\t{precision[1]:.3f}\t{recall[1]:.3f}\t{MAP[1]:.3f}\t{ndcg[1]:.3f}\t{hr[1]:.3f}')
        
        if not self.arg.debug:
            self.save_model(self.arg.out_path + 'saved_model.pt')
            
        if self.arg.plot or self.arg.debug:
            # plot all and save them in the out_path
            plt.figure()
            plt.plot(loss_list)
            plt.title('Loss')
            plt.savefig(self.arg.log_path + 'loss.png')
            
            plt.figure()
            plt.plot(precision_list)
            plt.title('Precision@10')
            plt.savefig(self.arg.log_path + 'precision@10.png')
            
            plt.figure()
            plt.plot(recall_list)
            plt.title('Recall@10')
            plt.savefig(self.arg.log_path + 'recall@10.png')
            
            plt.figure()
            plt.plot(MAP_list)
            plt.title('MAP@10')
            plt.savefig(self.arg.log_path + 'map@10.png')
            
            plt.figure()
            plt.plot(ndcg_list)
            plt.title('NDCG@10')
            plt.savefig(self.arg.log_path + 'ndcg@10.png')
            
            plt.figure()
            plt.plot(hr_list)
            plt.title('HR@10')
            plt.savefig(self.arg.log_path + 'hitrate@10.png')
            
            
            plt.figure()
            plt.plot(precision_list_20)
            plt.title('Precision@20')
            plt.savefig(self.arg.log_path + 'precision@20.png')
            
            plt.figure()
            plt.plot(recall_list_20)
            plt.title('Recall@10')
            plt.savefig(self.arg.log_path + 'recall@20.png')
            
            plt.figure()
            plt.plot(MAP_list_20)
            plt.title('MAP@20')
            plt.savefig(self.arg.log_path + 'map@20.png')
            
            plt.figure()
            plt.plot(ndcg_list_20)
            plt.title('NDCG@20')
            plt.savefig(self.arg.log_path + 'ndcg@20.png')
            
            plt.figure()
            plt.plot(hr_list_20)
            plt.title('HR@20')
            plt.savefig(self.arg.log_path + 'hitrate@20.png')
            
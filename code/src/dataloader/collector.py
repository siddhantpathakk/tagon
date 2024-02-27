import numpy as np
import random

class DataCollector:
    def __init__(self,config):
        self.arg = config
        self.file_path = self.arg.file_path
        
        self.L = self.arg.L
        self.H = self.arg.H
        self.N = 20
        self.topk = self.arg.topk

    def _load_raw_data_(self):
        self.data = open(self.file_path + 'u.data', 'rb').readlines()
        self.vname = open(self.file_path + 'u.vname', 'rb').readlines()
        self.demo = open(self.file_path + 'u.demo', 'rb').readlines()
        self.vcat = open(self.file_path + 'u.vcat', 'rb').readlines()
        
    def _demo2dict_(self):
        gender_tocken, age_tocken = 2, 3 
        onehot_len = gender_tocken * age_tocken 
        tocken2onehot,index = dict(),0
        for i in range(gender_tocken): 
            for k in range(age_tocken):
                tocken_ = str(i+1)+'-'+str(k+1)
                onehot_ = list()
                for h in range(onehot_len):
                    if h == index:
                        onehot_.append(1)
                    else:
                        onehot_.append(0)
                index += 1
                tocken2onehot[tocken_] = onehot_

        onehot_list = list(tocken2onehot.values())
        uid2attr = dict()
        for ny_demo_i in self.demo[1:]:
            ny_demo_i = ny_demo_i.decode('utf-8').split(',')
            uid_, age_, gender_ = ny_demo_i[0], int(ny_demo_i[1]), int(ny_demo_i[2])
            if age_ <= 25:
                age_ = 1
            elif  age_ > 25 and age_ <= 35:
                age_ = 2
            else:
                age_ = 3
            tocken_ = str(gender_) + '-' + str(age_)
            onehot_ = tocken2onehot[tocken_]
            if uid_ not in uid2attr:
                onehot_ = random.choice(onehot_list)
                uid2attr[uid_] = onehot_

        uid2attr['nothing'] = onehot_
        return uid2attr
    
    def _data2dict_(self):
        uid2locid_time,locid2detail = dict(),dict()
        for ny_checkin_i in self.data[1:]:
            ny_checkin_i = ny_checkin_i.decode('utf-8').split(',')
            uid_,locid_,rating_,time_ = ny_checkin_i[0],ny_checkin_i[1],ny_checkin_i[2],ny_checkin_i[3].strip('\n')
            if uid_ not in uid2locid_time:
                uid2locid_time[uid_] = list()
            uid2locid_time[uid_] = [(locid_,time_)] + uid2locid_time[uid_] 
            if locid_ not in locid2detail:
                locid2detail[locid_] = [rating_]
        
        return uid2locid_time,locid2detail
    
    def _vname2dict_(self):
        locid2locname = dict()
        for ny_vname_i in self.vname[1:]:
            ny_vname_i = ny_vname_i.decode('utf-8').split(',')
            locid_,locname_ = ny_vname_i[0],ny_vname_i[1]
            if locid_ not in locid2locname:
                locid2locname[locid_] = locname_
        return locid2locname
    
    def _vcat2dict_(self):   
        locid2catid = dict()
        for ny_vcat_i in self.vcat[1:]:
            ny_vcat_i = ny_vcat_i.decode('utf-8').split(',')
            locid_,catid_ = ny_vcat_i[0],str(ny_vcat_i[1]).strip('\n')
            if locid_ not in locid2catid:
                locid2catid[locid_] = catid_
        return locid2catid
   
    def _locid2detail_Merge_(self,locid2detail,locid2locname,locid2catid):
        locid_name_ = list(set(locid2detail.keys()))
        for i in range(len(locid_name_)):
            if locid_name_[i] in locid2locname:
                locname_ = locid2locname[locid_name_[i]]
                locid2detail[locid_name_[i]].append(locname_)
            else:
                locid2detail[locid_name_[i]].append('none_name')
            if locid_name_[i] in locid2catid:
                catid_ = locid2catid[locid_name_[i]]
                locid2detail[locid_name_[i]].append(catid_)
            else:
                locid2detail[locid_name_[i]].append('11')
        # print('locid2detail_merge = ',locid2detail)
        return locid2detail
    
    def main_data2dict(self):
        self._load_raw_data_()
        uid2locid_time,locid2detail = self._data2dict_()
        locid2locname = self._vname2dict_()
        locid2catid = self._vcat2dict_()
        uid2attr = self._demo2dict_()
        locid2detail = self._locid2detail_Merge_(locid2detail,locid2locname,locid2catid,uid2locid_time)
        return uid2locid_time,locid2detail,uid2attr

    def _limit_in_seqlen_(self,uid2locid_time):
        old_uid_list_ = list(uid2locid_time.keys())
        for i in range(len(old_uid_list_)):
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            if len(locid_time_) < (self.N + self.L + self.topk):
                del uid2locid_time[old_uid_list_[i]]
            else:
                uid2locid_time[old_uid_list_[i]] = uid2locid_time[old_uid_list_[i]][:self.N + self.L + self.topk]
        return uid2locid_time
     
    def _element2ids_(self,uid2locid_time,locid2detail):
        old_uid_list_ = list(uid2locid_time.keys()) 
        locid_list_,vc_list_ = list(),list()

        for i in range(len(old_uid_list_)):
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            for j in range(len(locid_time_)):
                locid_list_.append(locid_time_[j][0])
                detail_ = locid2detail[locid_time_[j][0]]
                vc_list_.append(detail_[-1])

        locid_list_ = list(set(locid_list_)) 
        vc_list_ = list(set(vc_list_)) 

        ids_ = 0
        self.uid2ids,self.locid2ids,self.vc2ids = dict(),dict(),dict() 

        for i in range(len(locid_list_)):
            self.locid2ids[locid_list_[i]] = ids_
            ids_ +=1
        
        for i in range(len(old_uid_list_)):
            self.uid2ids[old_uid_list_[i]] = ids_
            ids_ +=1
        
        for i in range(len(vc_list_)):
            self.vc2ids[vc_list_[i]] = ids_
            ids_ +=1  
        node_num = ids_ 
        return old_uid_list_,locid_list_,node_num

    def _replace_element_with_ids_(self,old_uid_list_,locid_list_,uid2locid_time,locid2detail,uid2attr):
        new_uid2locid_time = dict()
        for i in range(len(old_uid_list_)):
            uid_ids_ = self.uid2ids[old_uid_list_[i]]
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            new_uid2locid_time[uid_ids_] = [(self.locid2ids[locid_time_[j][0]],locid_time_[j][1]) for j in range(len(locid_time_))]
        new_locid2detail = dict()
        for i in range(len(locid_list_)):
            detail_ = locid2detail[locid_list_[i]]
            detail_[-1] = self.vc2ids[detail_[-1]]
            new_locid2detail[self.locid2ids[locid_list_[i]]] = detail_
        new_uid2attr = dict()
        for i in range(len(old_uid_list_)):
            uid_ids_ = self.uid2ids[old_uid_list_[i]]
            if old_uid_list_[i] in uid2attr:
                attr_ = uid2attr[old_uid_list_[i]]
            else:
                attr_ = uid2attr['nothing']  
            new_uid2attr[uid_ids_] = attr_
        return new_uid2locid_time,new_locid2detail,new_uid2attr

    def _seq_data_building_(self,old_uid_list_,uid2locid_time):
        new_uid_list_ = [self.uid2ids[old_uid_list_[i]] for i in range(len(old_uid_list_))]
        user_np,seq_train,seq_test,test_set = list(),list(),list(),list()
        
        for i in range(len(new_uid_list_)):
            locid_time_ = uid2locid_time[new_uid_list_[i]]
            train_part = locid_time_[ : self.N ] # first N items up for training
            testX_part = locid_time_[ self.N : self.N+self.L ] # next L items for validation
            testY_part = locid_time_[ self.N+self.L : ] # remaining items for testing

            # TODO: check if range should have +1 or not
            for j in range(len(train_part) - self.L - self.H + 1):
                train_part_j_ = train_part[j:j+self.L+self.H]
                user_np.append(new_uid_list_[i])
                seq_train.append([train_part_j_[k][0] for k in range(len(train_part_j_))])
            
            seq_test.append([testX_part[j][0] for j in range(len(testX_part))])
            test_set.append([testY_part[j][0] for j in range(len(testY_part))])
            
        user_np = np.array(user_np)
        seq_train = np.array(seq_train) # training set
        seq_test = np.array(seq_test) # validation set
        return new_uid_list_, user_np, seq_train, seq_test, test_set

    def _edge_building_(self,uid_list_,uid2locid_time,locid2detail):
        u2v,u2vc,v2u,v2vc = dict(),dict(),dict(),dict()
        for i in range(len(uid_list_)):
            locid_time_ = uid2locid_time[uid_list_[i]]
            v_list_, u_vc_list_ = list(),list()
            for j in range(len(locid_time_)):
                locid_ = locid_time_[j][0]
                v_list_.append(locid_)

                if locid_ not in v2u:
                    v2u[locid_] = list()
                v2u[locid_].append(uid_list_[i])

                vc_ = locid2detail[locid_][-1]
                u_vc_list_.append(vc_)
                if locid_ not in v2vc:
                    v2vc[locid_] = vc_
            v_list_ = list(set(v_list_))
            u2v[uid_list_[i]] = v_list_
            u_vc_list_ = list(set(u_vc_list_))
            u2vc[uid_list_[i]] = u_vc_list_
        v2u_keys = list(v2u.keys())
        for i in range(len(v2u_keys)):
            v2u[v2u_keys[i]] = list(set(v2u[v2u_keys[i]]))
        return u2v,u2vc,v2u,v2vc

    def main_datadict2traindata(self,uid2locid_time,locid2detail,uid2attr):
        # limit the length of the sequence
        uid2locid_time = self._limit_in_seqlen_(uid2locid_time)

        # convert and replace elements with ids
        old_uid_list_,locid_list_,node_num = self._element2ids_(uid2locid_time,locid2detail)
        uid2locid_time,locid2detail,uid2attr = self._replace_element_with_ids_(old_uid_list_,locid_list_,uid2locid_time,locid2detail,uid2attr)
        
        # build the sequence data
        uid_list_, user_np, seq_train, seq_test, test_set = self._seq_data_building_(old_uid_list_,uid2locid_time)
        
        # build the edge data
        u2v,u2vc,v2u,v2vc = self._edge_building_(uid_list_,uid2locid_time,locid2detail)
        
        relation_num =  4 #u_vc,u,v,v_vc
        
        return {
            'uid2locid_time': uid2locid_time,
            'locid2detail': locid2detail,
            'node_num': node_num,
            'relation_num': relation_num,
            'uid_list_': uid_list_,
            'user_np': user_np,
            'seq_train': seq_train,
            'seq_test': seq_test,
            'test_set': test_set,
            'u2v': u2v,
            'u2vc': u2vc,
            'v2u': v2u,
            'v2vc': v2vc
        }
        
    def main(self):
        # get train data
        uid2locid_time,locid2detail,uid2attr = self.main_data2dict()
        train_val_data = self.main_datadict2traindata(uid2locid_time,locid2detail,uid2attr)
        
        train_part = [train_val_data['user_np'], train_val_data['seq_train']]
        test_part = [train_val_data['seq_test'], train_val_data['test_set'], train_val_data['uid_list_'], train_val_data['uid2locid_time']]
        edges = [train_val_data['u2v'], train_val_data['u2vc'], train_val_data['v2u'], train_val_data['v2vc']]
        
        return train_part, test_part, edges, train_val_data['node_num'], train_val_data['relation_num']
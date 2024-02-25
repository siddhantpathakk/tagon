import pickle
import numpy as np
import random

class DataCollector:
    """
    The DataCollector class is designed for preprocessing and structuring data for a sequential recommendation system, specifically tailored to handle datasets with user interactions, demographic information, venue names, and categories. 
    """
    def __init__(self,config):
        """
        Initializes the DataCollector instance by setting up various configurations from the config argument. 
        
        These configurations include the file path to the dataset, parameters like L, H, topk for sequence lengths and recommendation settings, 
        and N as a predefined constant.

        Args:
            config (argparse.ArgumentParser): The configuration object containing the file path and other parameters.
        """
        self.arg = config
        self.file_path = self.arg.file_path
        
        self.L = self.arg.L
        self.H = self.arg.H
        self.topk = self.arg.topk
        
        self.N = 20

    def _load_raw_data_(self):
        """
        Loads the raw data files related to user interactions, venue names, user demographic information, and venue categories. Each data type is read from a separate file and stored in instance variables for further processing.
        """
        self.data = open(self.file_path + 'u.data', 'rb').readlines()
        self.vname = open(self.file_path + 'u.vname', 'rb').readlines()
        self.demo = open(self.file_path + 'u.demo', 'rb').readlines()
        self.vcat = open(self.file_path + 'u.vcat', 'rb').readlines()
        
    def _demo2dict_(self):
        """
        Converts demographic information into a dictionary where user IDs map to demographic attributes (like age and gender) encoded as one-hot vectors. This method also handles the categorization of age into predefined ranges and deals with missing values by assigning random attributes.

        Returns:
            dict: A dictionary where user IDs map to demographic attributes encoded as one-hot vectors.
        """
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
            uid_, age_, gender_ = ny_demo_i[0],int(ny_demo_i[1]),int(ny_demo_i[2])
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
        """
        Transforms user interaction data into dictionaries. One maps user IDs to lists of venue IDs and timestamps, indicating the sequence of interactions. The other maps venue IDs to details like ratings.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps.
        """
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
        """
        Creates a dictionary mapping venue IDs to their names, facilitating the association of venues with readable identifiers.


        Returns:
            dict: A dictionary where venue IDs map to venue names.
        """
        locid2locname = dict()
        for ny_vname_i in self.vname[1:]:
            ny_vname_i = ny_vname_i.decode('utf-8').split(',')
            locid_,locname_ = ny_vname_i[0],ny_vname_i[1]
            if locid_ not in locid2locname:
                locid2locname[locid_] = locname_
        return locid2locname
    
    def _vcat2dict_(self):   
        """
        Generates a dictionary mapping venue IDs to category IDs, allowing for the categorization of venues based on predefined types.
        

        Returns:
            dict: A dictionary where venue IDs map to category IDs.
        """
        locid2catid = dict()
        for ny_vcat_i in self.vcat[1:]:
            ny_vcat_i = ny_vcat_i.decode('utf-8').split(',')
            locid_,catid_ = ny_vcat_i[0],str(ny_vcat_i[1]).strip('\n')
            if locid_ not in locid2catid:
                locid2catid[locid_] = catid_
        return locid2catid
   
    def _locid2detail_Merge_(self,locid2detail,locid2locname,locid2catid,uid2locid_time):
        """
        Merges venue details, names, and categories into a single detailed structure. This method enhances the locid2detail dictionary with venue names and category IDs, enriching the venue-related information.


        Args:
            locid2detail (dict): A dictionary where venue IDs map to details like ratings.
            locid2locname (dict): A dictionary where venue IDs map to venue names.
            locid2catid (dict): A dictionary where venue IDs map to category IDs.
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.

        Returns:
            dict: A dictionary where venue IDs map to detailed information including names and category IDs.
        """
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
    
    def main_data2dict(self,save=True):
        """
        A high-level function that orchestrates the loading and processing of raw data into structured dictionaries for user interactions, venue details, and user attributes. Optionally, it saves the processed data into a pickle file for later use.


        Args:
            save (bool, optional): _description_. Defaults to True.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps.
            dict: A dictionary where venue IDs map to detailed information including names and category IDs.
            dict: A dictionary where user IDs map to demographic attributes encoded as one-hot vectors.
        """
        self._load_raw_data_()
        uid2locid_time,locid2detail = self._data2dict_()
        locid2locname = self._vname2dict_()
        locid2catid = self._vcat2dict_()
        uid2attr = self._demo2dict_()
        locid2detail = self._locid2detail_Merge_(locid2detail,locid2locname,locid2catid,uid2locid_time)
        return uid2locid_time,locid2detail,uid2attr

    def _limit_in_seqlen_(self,uid2locid_time):
        """
        Prunes user interaction sequences to ensure they meet the required sequence length defined by N, L, and topk. This step is crucial for maintaining consistency in input data for the recommendation model.


        Args:
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps, pruned to meet the required sequence length.
        """
        old_uid_list_ = list(uid2locid_time.keys())
        for i in range(len(old_uid_list_)):
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            if len(locid_time_) < (self.N + self.L + self.topk):
                del uid2locid_time[old_uid_list_[i]]
            else:
                uid2locid_time[old_uid_list_[i]] = uid2locid_time[old_uid_list_[i]][:self.N + self.L + self.topk]
        return uid2locid_time
     
    def _element2ids_(self,uid2locid_time,locid2detail):
        """
        Converts user and venue identifiers into a unified indexing system. This method prepares the data for numerical processing by replacing string IDs with numerical ones, facilitating easier handling in machine learning models.


        Args:
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.
            locid2detail (dict): A dictionary where venue IDs map to detailed information including names and category IDs.


        Returns:
            list: A list of user IDs.
            list: A list of venue IDs.
            int: The total number of nodes in the graph.
        """
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
        """
        Updates the user interaction data and venue details to use the new numerical IDs established by _element2ids_. This method ensures consistency in the representation of entities across the dataset.


        Args:
            old_uid_list_ (list): A list of user IDs.
            locid_list_ (list): A list of venue IDs.
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.
            locid2detail (dict): A dictionary where venue IDs map to detailed information including names and category IDs.
            uid2attr (dict): A dictionary where user IDs map to demographic attributes encoded as one-hot vectors.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps, using numerical IDs.
            dict: A dictionary where venue IDs map to detailed information including names and category IDs, using numerical IDs.
            dict: A dictionary where user IDs map to demographic attributes encoded as one-hot vectors.
        """
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
        """
        Constructs training and testing sequences from the interaction data. This method splits the data into sequences suitable for the sequential recommendation model, organizing them into inputs for training, validation, and testing phases.


        Args:
            old_uid_list_ (list): A list of user IDs.
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.
            
            
        Returns:
            list: A list of user IDs using numerical IDs.
        """
        new_uid_list_ = [self.uid2ids[old_uid_list_[i]] for i in range(len(old_uid_list_))]
        user_np,seq_train,seq_test,test_set = list(),list(),list(),list()

        for i in range(len(new_uid_list_)):
            locid_time_ = uid2locid_time[new_uid_list_[i]]
            train_part = locid_time_[:self.N] # first N items up for training
            testX_part = locid_time_[self.N:self.N+self.L] # next L items for val
            testY_part = locid_time_[self.N+self.L:] # next H items for testing

            for j in range(len(train_part)-self.L-self.H+1):
                train_part_j_ = train_part[j:j+self.L+self.H]
                user_np.append(new_uid_list_[i])
                seq_train.append([train_part_j_[k][0] for k in range(len(train_part_j_))])

            seq_test.append([testX_part[j][0] for j in range(len(testX_part))])
            test_set.append([testY_part[j][0] for j in range(len(testY_part))])
        user_np = np.array(user_np)
        seq_train = np.array(seq_train)
        seq_test = np.array(seq_test)
        return new_uid_list_,user_np,seq_train,seq_test,test_set

    def _edge_building_(self,uid_list_,uid2locid_time,locid2detail):
        """
        Builds edge relationships for graph-based models. It creates mappings between users and venues, and venues and categories, establishing the connections necessary for graph neural network processing.


        Args:
            uid_list_ (list): A list of user IDs using numerical IDs.
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.
            locid2detail (dict): A dictionary where venue IDs map to detailed information including names and category IDs.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs.
            dict: A dictionary where user IDs map to lists of venue categories.
            dict: A dictionary where venue IDs map to lists of user IDs.
            dict: A dictionary where venue IDs map to venue categories.
        """
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

    def main_datadict2traindata(self,uid2locid_time,locid2detail,uid2attr,save=True):
        """
        Converts the processed data dictionaries into structured training data formats, including sequences and edge relationships suitable for training graph-based sequential recommendation models. Optionally, it saves this structured data for later use.


        Args:
            uid2locid_time (dict): A dictionary where user IDs map to lists of venue IDs and timestamps.
            locid2detail (dict): A dictionary where venue IDs map to detailed information including names and category IDs.
            uid2attr (dict): A dictionary where user IDs map to demographic attributes encoded as one-hot vectors.
            save (bool, optional): _description_. Defaults to True.


        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps, using numerical IDs.
            dict: A dictionary where venue IDs map to detailed information including names and category IDs, using numerical IDs.
            int: The total number of nodes in the graph.
            int: The total number of relationships in the graph.
            list: A list of user IDs using numerical IDs.
            numpy.ndarray: A numpy array of user IDs.
            numpy.ndarray: A numpy array of training sequences.
            numpy.ndarray: A numpy array of testing sequences.
            numpy.ndarray: A numpy array of testing sets.
            dict: A dictionary where user IDs map to lists of venue IDs.
            dict: A dictionary where user IDs map to lists of venue categories.
            dict: A dictionary where venue IDs map to lists of user IDs.
            dict: A dictionary where venue IDs map to venue categories.
        """
        uid2locid_time = self._limit_in_seqlen_(uid2locid_time)

        old_uid_list_,locid_list_,node_num = self._element2ids_(uid2locid_time,locid2detail)

        uid2locid_time,locid2detail,uid2attr = self._replace_element_with_ids_(old_uid_list_,locid_list_,uid2locid_time,locid2detail,uid2attr)
        
        uid_list_,user_np,seq_train,seq_test,test_set = self._seq_data_building_(old_uid_list_,uid2locid_time)
        u2v,u2vc,v2u,v2vc = self._edge_building_(uid_list_,uid2locid_time,locid2detail)
        relation_num =  4 #u_vc,u,v,v_vc
        return uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc  

    def main(self,save1=True,save2=True):
        """
        The main entry point for the data preprocessing pipeline. It sequentially calls the methods to load raw data, process it into structured formats, and then transform it into training data ready for use in the recommendation model. It supports saving intermediate results to speed up future preprocessing steps.


        Args:
            save1 (bool, optional): Save the processed data into a pickle file for later use. Defaults to True.
            save2 (bool, optional): Save the training data into a pickle file for later use. Defaults to True.

        Returns:
            dict: A dictionary where user IDs map to lists of venue IDs and timestamps, using numerical IDs.
            dict: A dictionary where venue IDs map to detailed information including names and category IDs, using numerical IDs.
            int: The total number of nodes in the graph.
            int: The total number of relationships in the graph.
            list: A list of user IDs using numerical IDs.
            numpy.ndarray: A numpy array of user IDs.
            numpy.ndarray: A numpy array of training sequences.
            numpy.ndarray: A numpy array of testing sequences.
            numpy.ndarray: A numpy array of testing sets.
            dict: A dictionary where user IDs map to lists of venue IDs.
            dict: A dictionary where user IDs map to lists of venue categories.
            dict: A dictionary where venue IDs map to lists of user IDs.
            dict: A dictionary where venue IDs map to venue categories.
        """
        uid2locid_time,locid2detail,uid2attr = self.main_data2dict(save1)
        uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc = self.main_datadict2traindata(uid2locid_time,locid2detail,uid2attr,save2)
        return uid2locid_time, locid2detail, node_num, relation_num, uid_list_, \
            user_np, seq_train, \
            seq_test, test_set, \
            u2v, u2vc, v2u, v2vc 
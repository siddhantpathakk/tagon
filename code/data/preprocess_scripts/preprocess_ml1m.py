import os
import argparse
import numpy
from scipy.spatial.distance import hamming

class MovielensPreprocessor:
    def __init__(self, root_raw):
        self.root_raw = root_raw
    
    def load_raw_data(self):
        if not os.path.exists(self.root_raw):
            raise FileNotFoundError('Raw data not found at given location {}'.format(self.root_raw))
        
        self.u_data = open(self.root_raw + 'ratings.dat', 'rb').readlines()
        self.u_user = open(self.root_raw + 'users.dat', 'rb').readlines()
        self.u_item = open(self.root_raw + 'movies.dat', 'rb').readlines()
        
    def _binarize_ratings(self, rating, threshold_rating=4):
        if rating >= threshold_rating:
            return 1
        else:
            return 0
    
    def _convert_to_int(self, iterable):
        try:
            return [int(i) for i in iterable]
        except:
            raise ValueError('Cannot convert the given iterable to int')
      
    def _gender_to_int(self, gender):
        try:
            return 1 if gender == 'M' else 2
        except:
            raise AttributeError('Only two genders are supported: M and F') 
            
    def _data2dict(self):
        data_dict = dict()
        for row in self.u_data:
            row = row.decode('utf-8').split('::')
            user_id, item_id, rating, timestamp = row[0], row[1], row[2], row[3].strip()
            user_id, item_id, rating, timestamp = self._convert_to_int([user_id, item_id, rating, timestamp])
            rating = self._binarize_ratings(rating)
            if user_id not in data_dict:
                data_dict[user_id] = list()
            data_dict[user_id].append((item_id, rating, timestamp))
        return data_dict
    
    def _user2dict(self):
        user_dict = dict()
        for row in self.u_user:
            row = row.decode('utf-8').split('::')
            user_id, age, gender = row[0], row[2], row[1]
            user_id, age = self._convert_to_int([user_id, age])
            gender = self._gender_to_int(gender)
            if user_id not in user_dict:
                user_dict[user_id] = list()
                user_dict[user_id].append((age, gender))
        return user_dict

    def _calculate_similarity(self, vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError('Length of vectors are not equal')
        return hamming(vector1, vector2) * len(vector1)

    def _get_most_similar_genre(self, genre_vector):
        
        # ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        realism  = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
        fantasy  = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        family   = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        rom_com  = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        suspense = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]

        distances = [
            self._calculate_similarity(genre_vector, realism), # 0
            self._calculate_similarity(genre_vector, fantasy), # 1
            self._calculate_similarity(genre_vector, family), # 2
            self._calculate_similarity(genre_vector, rom_com), # 3
            self._calculate_similarity(genre_vector, suspense) # 4
        ]
        
        return numpy.argmin(distances) + 1
        
    def _item2dict(self):
        item_dict = dict()
        
        genre_init_vector = [0 for i in range(19)]
        genre_mapper = {
            'unknown': 0,
            'Action': 1,
            'Adventure': 2,
            'Animation': 3,
            "Children's": 4,
            'Comedy': 5,
            'Crime': 6,
            'Documentary': 7,
            'Drama': 8,
            'Fantasy': 9,
            'Film-Noir': 10,
            'Horror': 11,
            'Musical': 12,
            'Mystery': 13,
            'Romance': 14,
            'Sci-Fi': 15,
            'Thriller': 16,
            'War': 17,
            'Western': 18
        }
        
        for row in self.u_item:
            row = row.decode('latin-1').split('::')
            item_id, title = row[0], row[1]
            item_id = self._convert_to_int([item_id])[0]
            genre_vector_text = row[2].strip().split('|')
            genre_vector = genre_init_vector.copy()
            for genre in genre_vector_text:
                genre_vector[genre_mapper[genre]] = 1
            
            genre = None
            
            if len(genre_vector) == 0:
                genre = 0
            
            else:
                genre_vector = [int(i) for i in genre_vector]
                genre = self._get_most_similar_genre(genre_vector)
            
            if item_id not in item_dict:
                item_dict[item_id] = [title, genre]
                
        return item_dict
    
    def convert_to_dict(self, processed_root=None, save=False):
        data_dict = self._data2dict()
        user_dict = self._user2dict()
        item_dict = self._item2dict()
        
        if save and processed_root is not None:
            with open(processed_root + 'u.data', 'wb') as f:
                f.write('user_id,item_id,rating,timestamp'.encode('utf-8'))
                for key, value in data_dict.items():
                    for item in value:
                        f.write('\n{},{},{},{}'.format(key, item[0], item[1], item[2]).encode('utf-8'))
                    
            with open(processed_root + 'u.vname', 'wb') as f:
                f.write('item_id,title'.encode('utf-8'))
                for key, value in item_dict.items():
                    f.write('\n{},{}'.format(key, value[0]).encode('utf-8'))

            with open(processed_root + 'u.vcat', 'wb') as f:
                f.write('item_id,category'.encode('utf-8'))
                for key, value in item_dict.items():
                    f.write('\n{},{}'.format(key, value[1]).encode('utf-8'))

            with open(processed_root + 'u.demo', 'wb') as f:
                f.write('user_id,age,gender'.encode('utf-8'))
                for key, value in user_dict.items():
                    gender = 1 if value[0][1] == 'M' else 2
                    f.write('\n{},{},{}'.format(key, value[0][0], gender).encode('utf-8'))
                    
        elif save and processed_root is None:
            raise ValueError('Please provide a valid path to save the processed data')
    
        return data_dict, user_dict, item_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', '-d', type=str, default='./raw/ml-100k/')
    parser.add_argument('--processed', '-s', type=str, default='./processed/ml-100k/')
    args = parser.parse_args()

    data_preprocessor = MovielensPreprocessor(args.raw)
    data_preprocessor.load_raw_data()
    data_dict, user_dict, item_dict = data_preprocessor.convert_to_dict(args.processed, save=True)    
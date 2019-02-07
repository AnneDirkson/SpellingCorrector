
# coding: utf-8

# Spelling correction 
# 
# --author -- AR Dirkson 
# --date -- 5-2-2019 
# 
# This script is a spelling correction module that uses unsupervised data to construct a list of candidates. The correction algorithm is a weighted Levenshtein distance algorithm. A decision process is used to determine if a word is a spelling mistake.
# 
# It makes use of the CELEX generic dictionary but this can be substituted by another generic dictionary. It is only used to determine if a word should not be corrected because it is a generic word. 
# 
# The grid used for the spelling mistake detection was [0.05 - 0.15] (steps of 0.01) for relative weighted edit distance max and [2-10] (steps of 1) for relative corpus frequency multiplier. This can be re-tuned (tuning not included in this script).
# 
# Note: the damlev module only works on Linux platforms and the input data needs to be tokenized
# 

# In[3]:


from collections import Counter, defaultdict, OrderedDict
import editdistance
import re
import csv 
import pandas as pd
import pickle
from weighted_levenshtein import lev, osa, dam_lev

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import scipy.stats 

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


# In[36]:


class SpellingCorrector(): 
    
    def __init__(self): 
        pass
    
    def load_obj (self, path, name): 
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')
    
    def load_files (self): 
        #load the edit matrices
        path = '/data/dirksonar/Project1_lexnorm/spelling_correction/output/'
        #transpositions
        self.edits_trans = self.load_obj(path, 'weighted_edits_transpositions')
        #deletions 
        self.edits_del = self.load_obj(path,'weighted_edits_deletions')
        #insertions 
        self.edits_ins = self.load_obj(path,'weighted_edits_insertions')
        #substitutions
        self.edits_sub = self.load_obj(path,'weighted_edits_substitutions')
                
        #load the generic dictionary - CHANGE PATH!  
        self.celex_freq_dict = self.load_obj ('/home/dirksonar/Scripts/Project1_lexnorm/preprocessing_pipeline/obj_lex/', 'celex_lwrd_frequencies')
    
    
    def initialize_weighted_matrices(self): 
    #initialize the cost matrixes for deletions and insertions
        insert_costs = np.ones(128, dtype=np.float64)  # make an array of all 1's of size 128, the number of ASCII characters
        delete_costs = np.ones (128, dtype=np.float64)

        for index,row in self.edits_ins.iterrows(): 
            insert_costs[ord(index)] = row['transformed_frequency']

        for index,row in self.edits_del.iterrows(): 
            delete_costs[ord(index)] = row['transformed_frequency']

        #substitution

        substitute_costs = np.ones((128, 128), dtype=np.float64)
        lst = []
        for index,row in self.edits_sub.iterrows(): 
            z = tuple([row['edit_from'], row['edit_to'], row['transformed_frequency']])
            lst.append (z)
        for itm in lst: 
            itm2 = list(itm)
            try: 
                substitute_costs[ord(itm2[0]), ord(itm2[1])] = itm2[2]
            except IndexError: 
                pass

        #transposition

        transpose_costs = np.ones((128, 128), dtype=np.float64)

        lst = []

        for index,row in self.edits_trans.iterrows(): 
            z = tuple([row['first_letter'], row['second_letter'], row['transformed_frequency']])
            lst.append (z)

        for itm in lst: 
            itm2 = list(itm)
            try: 
                transpose_costs[ord(itm2[0]), ord(itm2[1])] = itm2[2]
            except IndexError: 
                print(itm2)

        return insert_costs, delete_costs, substitute_costs, transpose_costs

    
    def weighted_ed_rel (self, cand, token, del_costs, ins_costs, sub_costs, trans_costs): 
        w_editdist = dam_lev(token, cand, delete_costs = del_costs, insert_costs = ins_costs, 
                             substitute_costs = sub_costs, transpose_costs = trans_costs)
        rel_w_editdist = w_editdist/len(token)
        return rel_w_editdist

    def run_low (self, word, voc, func, del_costs, ins_costs, sub_costs, trans_costs): 
        replacement = [' ',100]
        for token in voc: 
            sim = func(word, token, del_costs, ins_costs, sub_costs, trans_costs)
            if sim < replacement[1]:
                replacement[1] = sim
                replacement[0] = token

        return replacement   
    
    
    def spelling_correction (self, post, token_freq_dict, token_freq_ordered, min_rel_freq = 2, max_rel_edit_dist = 0.08): 
        post2 = []
        cnt = 0 

        for a, token in enumerate (post): 
            if self.TRUE_WORD.fullmatch(token):
                if token in self.spelling_corrections:
                    correct = self.spelling_corrections[token] 
                    post2.append(correct)
                    cnt +=1
                    self.replaced.append(token)
                    self.replaced_with.append(correct)

                elif token in self.celex_freq_dict:
                    post2.append(token)

                else:

                    # make the subset of possible candidates
                    freq_word = token_freq_dict[token]
                    limit = freq_word * min_rel_freq
                    subset = [t[0] for t in token_freq_ordered if t[1]>= limit]

                    #compare these candidates with the word        
                    candidate = self.run_low (token, subset, self.weighted_ed_rel, self.delete_costs_nw, self.insert_costs_nw, 
                                         self.substitute_costs_nw, self.transpose_costs_nw)

                #if low enough RE - candidate is deemed good
                    if candidate[1] <= max_rel_edit_dist:
                        post2.append(candidate[0]) 
                        cnt +=1
                        self.replaced.append(token)
                        self.replaced_with.append(candidate[0])
                        self.spelling_corrections [token] = candidate[0]
                    else: 
                        post2.append(token)
            else: post2.append(token)
        self.total_cnt.append (cnt)
        return post2
      
    def initialize_files_for_spelling(self): 
        total_cnt = []
        replaced = []
        replaced_with = []
        spelling_corrections= {}
        return total_cnt, replaced, replaced_with, spelling_corrections
    
    def change_tup_to_list (self, tup): 
        thelist = list(tup)
        return thelist

    def create_token_freq (self, data): 
        flat_data = [item for sublist in data for item in sublist]
        self.token_freq = Counter(flat_data)
        
        token_freq_ordered = self.token_freq.most_common ()
        self.token_freq_ordered2 = [self.change_tup_to_list(m) for m in token_freq_ordered]
    
    def correct_spelling_mistakes(self, data): 
#         data= self.load_obj ('/data/dirksonar/Project1_lexnorm/spelling_correction/output/', 'gistdata_lemmatised')
        self.load_files()
        self.insert_costs_nw, self.delete_costs_nw, self.substitute_costs_nw, self.transpose_costs_nw = self.initialize_weighted_matrices()
        self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections = self.initialize_files_for_spelling()
    
        self.TRUE_WORD = re.compile('[-a-z]+')  # Only letters and dashes  
#         data2 = [word_tokenize(m) for m in data]
        self.create_token_freq(data)
        out = [self.spelling_correction (m, self.token_freq, self.token_freq_ordered2) for m in data]
        return out, self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections


# In[37]:


#example script for running class 
out, total_cnt, replaced, replaced_with, spelling_corrections = SpellingCorrector().correct_spelling_mistakes(data)


# In[38]:


print(sum(total_cnt))


# In[40]:


c = Counter(replaced)
print(c.most_common (10))


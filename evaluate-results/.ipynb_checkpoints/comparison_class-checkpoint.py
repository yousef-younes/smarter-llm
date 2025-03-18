

import evaluation_measures as em

import pdb

class Compare_two_json_objects:
    def __init__(self):
        self.pred = None
        self.ground_truth = None
        
        self.counter = 0

        #Each of these lists contain the results of five measures
        self.title_scores = [0.0]*6
        self.author_name_scores = [0.0]*6
        self.author_affiliation_scores = [0.0]*6
        self.author_email_scores = [0.0]*6

        self.eval_obj = em.MeasureClass()
        
    def compare_titles(self,pred_title,gt_title):
        self.eval_obj.prediction = "" if pred_title == 'NA' else pred_title

        self.eval_obj.ground_truth = gt_title
        
        temp_scores = self.eval_obj.compare_using_all_measures();
    
        for i,x in enumerate(temp_scores):
            self.title_scores[i] += x
    
    def compare_author_names(self,pred_name,gt_name):
        self.eval_obj.prediction = "" if pred_name == 'NA' else pred_name
        self.eval_obj.ground_truth = gt_name
        
        temp_scores = self.eval_obj.compare_using_all_measures()
    
        for i,x in enumerate(temp_scores):
            self.author_name_scores[i] +=x
    
    def compare_author_emails(self,pred_email,gt_email):
        self.eval_obj.prediction = "" if pred_email == None else pred_email
        self.eval_obj.ground_truth = gt_email
        temp_scores = self.eval_obj.compare_using_all_measures()
    
        for i,x in enumerate(temp_scores):
            self.author_email_scores[i] +=x
    
    def compare_author_affiliation(self,pred_aff,gt_aff):
        temp_scores = None
        #handle when number of affliations between predicted and ground truth is not similar
        if len(pred_aff) < len(gt_aff):
            pred_aff.extend([""]*(len(gt_aff)-len(pred_aff)))
        else:
            pred_aff=pred_aff[:len(gt_aff)]
            
        for x,y in zip(pred_aff,gt_aff):
            self.eval_obj.prediction = "" if x=='NA' else x
            self.eval_obj.ground_truth = y 
            temp_scores = self.eval_obj.compare_using_all_measures()

        if temp_scores == None:
            pdb.set_trace()
        for i,x in enumerate(temp_scores):
            self.author_affiliation_scores[i] +=x
        
    
    def compare_authors(self,pred_authors,gt_authors):
        if len(pred_authors) != len(gt_authors):
            if len(pred_authors) < len(gt_authors): #if number of predicted authors is less than ground truth authors
                pred_authors.extend([{'name': 'NA', 'affiliations': ['NA'], 'email': 'NA'}]*(len(gt_authors)-len(pred_authors)))
            else:
                pred_authors = pred_authors[:len(gt_authors)] 
        
        for i in range(len(gt_authors)): #loop through all authors
            gt_author = gt_authors[i]  #assume the authors order is the same in gt ant pred lists
            gt_author_name = gt_author['name']
            gt_author_affiliations = gt_author['affiliations']
            gt_author_email = gt_author['email']
            
            pred_author = pred_authors[i]
            if isinstance(pred_author,dict) and isinstance(gt_author,dict):
                if "name" in pred_author:
                    pred_name = pred_author['name']
                else:
                    pred_name = ""
    
                self.compare_author_names(pred_name, gt_author_name)
    
                if "affiliations" in pred_author:
                    pred_aff_list = pred_author['affiliations']
                else:
                    pred_aff_list = []

                self.compare_author_affiliation(pred_aff_list,gt_author_affiliations)
    
                if "email" in pred_author:
                    pred_email = pred_author['email']
                else:
                    pred_email = ""
    
                self.compare_author_emails(pred_email, gt_author_email)
        
        
    def deep_compare(self):
        self.counter +=1 #increment the object counter
        
        gt_title = self.ground_truth['title']
        gt_authors = self.ground_truth['authors']
        
        if isinstance(self.pred,dict) and isinstance(self.ground_truth,dict):
            if "title" in self.pred:
                pred_title = self.pred['title']
            else:
                pred_title = ""
    
            self.compare_titles(pred_title, gt_title)
                
            if "authors" in self.pred:
                pred_authors = self.pred["authors"]
            else:
                pred_authors = []

            self.compare_authors(pred_authors,gt_authors)

    '''
    This function divides the obtained scores for all meta data on the number of instances to get the average
    '''
    def get_avg_scores(self):
        avg_title_scores = [e/self.counter for e in self.title_scores]
        avg_author_name_scores = [e/self.counter for e in self.author_name_scores]
        avg_author_affiliation_scores = [e/self.counter for e in self.author_affiliation_scores]
        avg_author_email_scores = [e/self.counter for e in self.author_email_scores]

        return avg_title_scores, avg_author_name_scores, avg_author_affiliation_scores,avg_author_email_scores
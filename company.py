import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.data import find
from nltk.stem import WordNetLemmatizer,PorterStemmer
from transformers import pipeline
import pandas as pd


# For fastness create a singleton class
class OpenTextHandler:
    __handler=None
    
    @staticmethod
    def get_instance():
        if OpenTextHandler.__handler is None:
            OpenTextHandler.__handler=OpenTextHandler()
        return OpenTextHandler.__handler
            
    def __init__(self) -> None:
        # Prerequisites
        for resource in ['tokenizers/punkt','corpora/stopwords']:
            try:
                find(resource)
                print(f'[INFO]Resource {resource} is already downloaded')
            except LookupError:
                print('[INFO]Resource {resource} not found. Downloading ...')
                nltk.download(resource.strip().split('/')[1])
        
        self.company_sector_categories=['tech','innovation','biomed','other']
        self.classifier=pipeline(task='text-classification',model='distilbert-base-uncased-finetuned-sst-2-english')      
        self.stopwords = set(stopwords.words('english'))
        self.stemming=PorterStemmer()
        self.lemmatizer=WordNetLemmatizer()

        
        
        self.categories = {
            'tech': [
                'technology', 'software', 'hardware', 'IT', 'cloud', 'AI', 'machine learning', 
                'computing', 'coding', 'programming', 'app', 'application', 'devops', 'agile', 'scrum',
                'computer', 'chip', 'semiconductor', 'processor', 'device', 'cloud computing', 
                'cloud services', 'SaaS', 'IaaS', 'PaaS', 'data center', 'virtualization', 
                'artificial intelligence', 'neural networks', 'deep learning', 'data science', 'algorithm', 
                'automation', 'robotics', 'networking', 'network', 'internet', 'connectivity', 'IoT', 
                'internet of things', 'cybersecurity', 'security', 'encryption', 'firewall', 'malware', 
                'virus', 'threat detection', 'emerging tech', 'blockchain', 'fintech', 'quantum computing', 
                'augmented reality', 'virtual reality', '5G'
            ],
            'innovation': [
                'innovative', 'invention', 'creative', 'disruptive', 'startup', 'innovation', 
                'creativity', 'design thinking', 'ideation', 'brainstorming', 'concept development', 
                'entrepreneurship', 'venture', 'incubation', 'accelerator', 'pitch', 'fundraising', 
                'game-changing', 'paradigm shift', 'breakthrough', 'revolutionary', 'next-gen', 
                'future tech', 'R&D', 'research', 'innovation lab', 'prototyping', 'experimentation', 
                'testing', 'technology transfer', 'commercialization', 'tech transfer', 'patent', 
                'intellectual property', 'licensing'
            ],
            'biomed': [
                'biomedical', 'healthcare', 'medical', 'biotechnology', 'clinical', 'biomed', 
                'health', 'medicine', 'patient care', 'hospital', 'clinic', 'research', 'lab', 
                'laboratory', 'bioscience', 'life sciences', 'molecular biology', 'pharma', 
                'pharmaceutical', 'drug development', 'medication', 'vaccine', 'therapeutic', 
                'medical device', 'diagnostics', 'imaging', 'surgical', 'equipment', 'instrumentation', 
                'clinical trial', 'clinical research', 'study', 'trial', 'participant', 'protocol', 
                'investigator', 'health IT', 'electronic health records', 'EHR', 'telemedicine', 
                'health informatics', 'mHealth', 'regulatory', 'compliance', 'FDA', 'approval', 
                'certification', 'standards', 'guidelines'
            ],
        }
        self.categories={category:[value.lower().strip() for value in values] for category,values in self.categories.items()}

    def preprocess_text(self,text):
        sentence_tokenization=sent_tokenize(text)
        filtered_tokens=[]
        for stoken in sentence_tokenization:
            print(stoken)
            word_tokens=word_tokenize(stoken.lower())
            generate_tokens=[self.stemming.stem(token) for token in word_tokens if token not in self.stopwords and token.isalpha()]
            generate_tokens=[self.lemmatizer.lemmatize(token) for token in word_tokens if token not in self.stopwords and token.isalpha()]
            filtered_tokens.extend(generate_tokens) 
        return filtered_tokens

    def categorize_company_sector(self,text):
        tokens=self.preprocess_text(text)
        matched_categories={category:0 for category in self.categories}
        for token in tokens:
            for category,keywords in self.categories.items():
                if token in keywords:
                    matched_categories[category]+=1
        
        best_category='other'
        max_matches=0
        for category,matches in matched_categories.items():
            if matches>max_matches:
                max_matches=matches
                best_category=category
                
        score=None
        best_category=best_category.replace("'","")
        if best_category=='tech':
            score=8
        elif best_category=='innovation':
            score=9
        elif best_category=='biomed':
            score=8
        else:
            score=7
        return (best_category,score)

    def categorize_companys_cause(self,companys_cause_statement):
        result=self.classifier(companys_cause_statement)
        label=result[0]['label']
        
        if label=="POSITIVE":
            return 9
        elif label=="NEGATIVE":
            return 0
        else:
            return -1
    
    def market_demand_supply_relation(self,statement):
        result=self.classifier(statement)
        label=result[0]['label']
        
        if label=="POSITIVE":
            return 8
        elif label=="NEGATIVE":
            return 3
        else:
            return -1
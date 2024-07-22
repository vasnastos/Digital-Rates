import pandas as pd
from geopy.geocoders import Nominatim
import pycountry_convert as pc
from textblob import TextBlob
from transformers import pipeline
from company import OpenTextHandler
from transformers import AutoTokenizer,AutoModelForCausalLM
from faker import Faker
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def get_country_info_from_postal_code(postal_code):
    geolocator=Nominatim(user_agent='GlobeOneDigitalPostalAPIv1')
    location=geolocator.geocode({"postalcode":postal_code},addressdetails=True)
    
    if location and 'address' in location.raw:
        address=location.raw['address']
        city=address.get('city',address.get('town',address.get('village',None)))
        country=address.get('country',None)
        
        try:
            if country=="Ελλάς":
                country="Greece"
            country_code=pc.country_name_to_country_alpha2(country)
            continent_code=pc.country_alpha2_to_continent_code(country_code)
            continent=pc.convert_continent_code_to_continent_name(continent_code)
        except KeyError:
            continent=None
        
        return {
            'city':city.lower(),
            'country':country.lower(),
            'continent':continent.lower()
        }
    else:
        raise ValueError(f"Location details not found for {postal_code}")

class SentimentAnalyzer:
    def __init__(self,method):
        self.sentiment_analysis_method=method
        if self.sentiment_analysis_method.upper()=="BERT":
            self.bert_model=pipeline('sentiment-analysis',model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    def __call__(self,open_text:str):
        if self.sentiment_analysis_method=="textblob":
            blob=TextBlob(text=open_text)
            polarity=blob.sentiment.polarity
            
            if polarity>0:
                return 'positive'
            elif polarity<0:
                return 'negative'
            else:
                return "neutral"
        elif self.sentiment_analysis_method.upper()=="BERT":
            if not hasattr(self,"bert_model"):
                raise ModuleNotFoundError("Bert model not supported")
            results=self.bert_model(open_text)
            for result in results:
                label=result['label']
                score=result['score']
                
                if label in ['LABEL_3','LABEL_4']:
                    sentiment='positive'
                elif label=='LABEL_2':
                    sentiment='neutral'
                else:
                    sentiment='negative'
            
            return sentiment
    
class Scoring:
    def __init__(self,default=True):
        self.score_funcs=list()
        self.sentiment_analyzer=SentimentAnalyzer(method="BERT")
        if default:
            self.apply_default_scoring_funcs()

    def score_from_postal_code(self,postal_code):
        postal_code_info=get_country_info_from_postal_code(postal_code)
        if postal_code_info['country']=='united states':
            return 9
        elif postal_code_info['country']=='Greece':
            if postal_code_info['city']=='athens':
                return 7
            else:
                return 3
        elif postal_code_info['continent']=='europe':
            return 8
        else:
            return 4  
    
    def score_from_sentiment_analysis(self,open_text:str):
        outcome=self.sentiment_analyzer(open_text)
        return 0 if outcome=='negative' else 8 if outcome=='positive' else 5
    
    def create_score_funcs(self):
        self.score_funcs.append(lambda company_name:10 if len(company_name)!=0 and company_name is not None else 0)
        self.score_funcs.append(self.score_from_postal_code)
        self.score_funcs.append(lambda shareholder_name: 10 if len(shareholder_name)!=0 and shareholder_name is not None else 0)
        self.score_funcs.append(lambda ceo_name:10 if len(ceo_name)!=0 and ceo_name is not None else 0)
        self.score_funcs.append(None) # Add function of years of establishment
        self.score_funcs.append(lambda turnover:10 if turnover>1e6 else 7)
        self.score_funcs.append(lambda years_of_establishment:int(years_of_establishment)+5 if years_of_establishment<5 else 10)
        self.score_funcs.append(lambda successful_rounds_of_funding:int(successful_rounds_of_funding)+5 if successful_rounds_of_funding<5 else 10)
        self.score_funcs.append(None)
        self.score_funcs.append(None)
        self.score_funcs.append(lambda total_funding:6 if total_funding>=int(5e4) and total_funding<int(1e5) else 7 if total_funding>=int(1e5) and total_funding<int(25e4) else 8 if total_funding>=int(25e4) and total_funding<int(5e5) else 9 if total_funding>=int(5e5) and total_funding<int(1e6) else 10)
        self.score_funcs.append(lambda company_sector:OpenTextHandler.get_instance().categorize_companys_sector(company_sector)[1])
        self.score_funcs.extend([None,None,None,None])
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(lambda companys_strategy:4 if companys_strategy.startswith('a') else 7)
        self.score_funcs.append(lambda companys_approach_to_competition:4 if companys_approach_to_competition.startswith('a') else 7)
        self.score_funcs.append(lambda companys_just_cause:OpenTextHandler.get_instance().categorize_companys_cause(companys_just_cause))
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(self.score_from_sentiment_analysis)
        self.score_funcs.append(lambda company_statement_selector:4 if company_statement_selector.startswith('a.') else 8)
        self.score_funcs.append(lambda employees_sentiment_status:6 if employees_sentiment_status.startswith('a.') else 7)
        self.score_funcs.append(lambda product_and_org_aesthetics:9 if product_and_org_aesthetics.startswith('a.') else 4)
        self.score_funcs.append(lambda org_external_influence:4 if org_external_influence.startswith('a') else 5 if org_external_influence.startswith('b') else 9)
        self.score_funcs.append(lambda org_market_set:10 if org_market_set.startswith('a.') else 7 if org_market_set.startswith('b.') else 4 if org_market_set.startswith('c.') else 2)
        self.score_funcs.append(OpenTextHandler.get_instance().market_demand_supply_relation)
        self.score_funcs.append(lambda org_organizational_ability:10 if org_organizational_ability.startswith('a.') else 7 if org_organizational_ability.startswith('b.') else 3 if org_organizational_ability.startswith('c.') else 4)
        self.score_funcs.append(self.score_from_sentiment_analysis)

turnover_params = {
    'tech': (6, 1.0),  # Mean and sigma for log-normal distribution
    'innovation': (5, 1.0),
    'biomed': (5.5, 1.0),
    'other': (4, 1.0)
}

description_prompts = {
    'tech': 'Our company specializes in developing cutting-edge technology solutions such as',
    'innovation': 'We are pioneers in creating innovative products and services including',
    'biomed': 'Our focus is on advancing biomedical research and providing healthcare solutions like',
    'other': 'We provide a range of services including'
}


class Questionnarie:
    def __init__(self,questionnaire_file:str,save_new_format=False) -> None:
        self.questions=pd.read_excel(questionnaire_file,engine='openpyxl',index_col=None,skiprows=2)
        self.questions_container=dict()
        
        options_buffer=list()
        current_question_index=None
        for index,row in self.questions.iterrows():
            text=row["QUESTIONS"].strip()
            
            if text.startswith(('a.','b.','c.','d.')):
                options_buffer.append(text)
            else:
                if options_buffer and current_question_index is not None:
                    self.questions_container[self.questions.at[current_question_index,"QUESTIONS"]]=options_buffer.copy()
                    options_buffer.clear()
                if row['ADDITIONAL attachements or info'].strip()=="yes/no":
                    self.questions_container[self.questions.at[index,"QUESTIONS"]]=["yes","no"]
                current_question_index=index
        
        if options_buffer and current_question_index is not None:
            self.questions_container[self.questions.at[current_question_index,"QUESTIONS"]]=options_buffer
    
        if save_new_format:
            with pd.ExcelWriter('./data/questionnaire.xlsx',engine='openpyxl') as writer:
                self.questions.to_excel(excel_writer=writer,index=False)

        self.fake=Faker()
        self.company_categories=['tech','innovation','biomed','other']
        self.tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        self.text_generator=AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-large"
        )
        
         
    def calculate_years_of_operation(self,year_of_establishment):
        current_year=2024
        if random.choice([True,False]):
            return current_year-year_of_establishment
        else:
            end_year=random.randint(year_of_establishment,current_year)
            return end_year-year_of_establishment
    
    def generate_text(self,prompt,max_length=150):
        input_ids=self.tokenizer(prompt,return_tensors="pt")
        outputs=self.text_generator.generate(**input_ids,penalty_alpha=0.6, top_k=4, max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)
    
    def generate_founding_rounds(self,turnover,years_of_operations):
        mean_funding=(turnover/50)+(years_of_operations/10)
        random_factor=np.random.normal(loc=0,scale=1)
        funding_rounds=np.random.normal(loc=mean_funding+random_factor,scale=2)
        funding_rounds=max(1,min(5,int(round(funding_rounds))))
        return funding_rounds
    
    def generate_fake_links(self,company_name):
        social_media_platforms = ['twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com']
        if random.choice([True,False]):
            return {
            'Twitter': f'https://{social_media_platforms[0]}/{company_name.replace(" ","-")}',
            'Facebook': f'https://{social_media_platforms[1]}/{company_name.replace(" ","-")}',
            'LinkedIn': f'https://{social_media_platforms[2]}/in/{company_name.replace(" ","-")}',
            'Instagram': f'https://{social_media_platforms[3]}/{company_name.replace(" ","-")}',
            'Press-Link':self.fake.url()
            }
        return None
    
    def generate_yes_no_answer(self,prompt):
        just_cause=np.random.randint(1,2)
        if just_cause==1:
            return ("yes",self.generate_text(prompt))
        else:
            return ("no",None)
        
    def generate_company(self):
        company_name=self.fake.company()
        company_sector=self.company_categories[random.randint(0,len(self.company_categories)-1)]
        mean,sigma=turnover_params[company_sector]
        year_of_establishment=random.randint(1985,2024)
        company_description=self.generate_text(prompt=description_prompts[company_sector])
        turnover=round(np.random.lognormal(mean,sigma))
        years_of_operation=self.calculate_years_of_operation(year_of_establishment)
        product_service=self.generate_text(prompt=f"Company's description:{company_description}. What is the company's product/service and how it's solving problems?")
        return {
            "Q1-name":company_name,
            "Q2-category":company_sector,
            "Q3-postal_code":self.fake.postalcode(),
            "Q4-Main Shareholder":self.fake.name_male()+" "+self.fake.last_name() if random.randint(1,2)==1 else self.fake.name_female()+" "+self.fake.last_name_female(),
            "Q5-ceo_name":self.fake.name_male()+" "+self.fake.last_name() if random.randint(1,2)==1 else self.fake.name_female()+" "+self.fake.last_name_female(),
            "Q6-year_of_establishment":year_of_establishment,
            "Q7-turnover-2021":turnover,
            "Q8-years_of_operation":years_of_operation,
            "Q9-founding_rounds":self.generate_founding_rounds(turnover,years_of_operation),
            "Q10-equipment":self.generate_text(prompt=f"The needed equipment for {company_name} categorized  in {company_sector} sector will be"),
            "Q11-equipement_cost":np.random.uniform(10000,200000),
            "Q12-company_sector":(company_description,company_sector),
            "Q13-company_links":self.generate_fake_links(company_name),
            "Q14-General_terms_of_renting":np.random.randint(1,10),
            "Q15-Other Considarations":self.generate_text(prompt=f'{company_description}\nIs there anything else we should consider?',max_length=150),
            "Q16-email":f'contact@{company_name.replace(" ","")}.com',
            "Q17-mission_statement":self.generate_text(prompt=f"Company's description:{company_description}. The company's mission statement is "),
            "Q18(product/service)":product_service,
            "Q19":"a" if np.random.randint(1,2)==1 else "b",
            "Q20":"b" if np.random.randint(1,2)==1 else "b",
            "Q21":self.generate_yes_no_answer(prompt=f'Company description:{company_description}\nWhich is the just cause of the company?'),
            "Q22":self.generate_text(prompt=f"Company product:{product_service}\n Is your product/service proposition inclusive for other societal groups? And in which and how?"), 
            "Q23":self.generate_text(prompt=f"Company description:{company_description}\n What is your moral foundation of the company? Why did the company enter this business?"),
            "Q24":self.generate_text(prompt=f"Company description:{company_description}\n What is your vision? How is it embedded in your organizational strategy?"),
            "Q25":self.generate_text(prompt=f"Company description:{company_description}\n What is your market? How is it defined?"),    
            "Q26":self.generate_text(prompt=f"Company description:{company_description}\n Product service:{product_service}\n Give me a short description of which is the company\'s core technology advantage")
        }
        
    
if __name__=='__main__':
    qn=Questionnarie(questionnaire_file='./data/qnnaire.xlsx')
    print(qn.generate_company())
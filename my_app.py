####################### INSTALL ##################################

import json
import os

#this should be upadted if the license key is different than 4.2.8.
with open('/content/4.2.8.spark_nlp_for_healthcare.json') as f:
    license_keys = json.load(f)

for k,v in license_keys.items(): 
    # %set_env $k=$v
    os.environ[k] = v

def start(secret):
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:"+version) \
        .config("spark.jars", "https://pypi.johnsnowlabs.com/"+secret+"/spark-nlp-jsl-"+jsl_version+".jar")
      
    return builder.getOrCreate()


from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline
from sparknlp_display import NerVisualizer
from sparknlp_display import RelationExtractionVisualizer

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp
import pandas as pd

import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

import spacy
from spacy import displacy

params = {"spark.driver.memory":"16G",
"spark.kryoserializer.buffer.max":"2000M",
"spark.driver.maxResultSize":"2000M"}

spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)

print ("Spark NLP Version :", sparknlp.version())
print ("Spark NLP_JSL Version :", sparknlp_jsl.version())

##################################################################

#if st.get_option("theme.primaryColor") is None:
#    # Use a default light theme
#    st.set_page_config(layout="wide")
#
#else:
#    # Use a dark theme
#    st.set_page_config(layout="wide",
#        initial_sidebar_state="expanded",
#        theme="dark"       
#    )

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='ADE Drug_Brandname Relation',  # String or None. Strings get appended with "• Streamlit". 
    page_icon= '/content/favicon.png',  # String, anything supported by st.image, or None.
)


st.sidebar.image('/content/jsl-logo.png', use_column_width=True)
st.sidebar.title('NER Models and Visualizers')


html_temp = """
<div style="background-color:Navy;padding:0px">
<h2 style="color:SkyBlue;text-align:center;">NER Playground for Drug Brandname - ADE Entities </h2>
</div>"""

st.markdown(html_temp, unsafe_allow_html=True)
st.image('/content/drug_ade.png', use_column_width=True)



#################################

#data
clinical_text1 = """I have an allergic reaction to vancomycin so I have itchy skin, sore throat/burning/itching, numbness of tongue and gums. I would not recommend this drug to anyone, especially since I have never had such an adverse reaction to any other medication."""
clinical_text2 = """Always tired, and possible blood clots. I was on Voltaren for about 4 years and all of the sudden had a minor stroke and had blood clots that traveled to my eye. I had every test in the book done at the hospital, and they couldnt find anything. I was completley healthy! I am thinking it was from the voltaren. I have been off of the drug for 8 months now, and have never felt better. I started eating healthy and working out and that has help alot. I can now sleep all thru the night. I wont take this again. If I have the back pain, I will pop a tylonol instead."""
clinical_text3 = """I experienced fatigue, muscle cramps, anxiety, agression and sadness after taking Lipitor but no more adverse after passing Zocor."""
clinical_text4 = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge."""
clinical_text5 = """History of present illness: The patient was recently started on a new medication for hypertension, amlodipine 5mg once daily, by her primary care physician. Within 3 days of starting the medication, she developed nausea and vomiting. She stopped taking the medication and the symptoms improved. She was subsequently referred to the emergency department for further evaluation."""
clinical_text6 = """I have Rhuematoid Arthritis for 35 yrs and have been on many arthritis meds. 
I currently am on Relefen for inflamation, Prednisone 5mg, every other day and Enbrel injections once a week. 
I have no problems from these drugs. Eight months ago, another doctor put me on Lipitor 10mg daily because my chol was 240. 
Over a period of 6 months, it went down to 159, which was great, BUT I started having terrible aching pain in my arms about that time which was radiating down my arms from my shoulder to my hands.
"""
clinical_text7 = """ Hypersensitivity to aspirin can be manifested as acute asthma, urticaria and/or angioedema, or a systemic anaphylactoid reaction.
A patient had undergone a renal transplantation as a result of malignant hypertension, and immunosuppressive therapy consisting of cyclosporin and prednisone ,  developed  sweating  and  thrombosis alone 5 years following the transplantation but there were not stomach pain.  
A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for   rheumatoid arthritis  presented  with  tense bullae and cutaneous fragility on the face and the back of the hands."""
clinical_text8 = """We describe the side effects of 5-FU in a colon cancer patient who suffered severe mucositis,  prolonged myelosuppression, and neurologic toxicity that required admission to the intensive care unit who  has a healthy appetite.
The reported cases of in utero exposure to cyclosposphamide shared the following manifestations with our patient who suffered  growth deficiency, developmental delay, craniosynostosis, blepharophimosis, flat nasal bridge and abnormal ears.
I have an allergic reaction to vancomycin so I have itchy skin, sore throat/burning/itching, numbness of tongue and gums.I would not recommend this drug to anyone, especially since I have never had such an adverse reaction to any other medication.
I experienced fatigue, muscle cramps, anxiety, agression and sadness after taking Lipitor but no more adverse after passing Zocor.
A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."""

################## PIPELINE ###############

model_name = ["ner_posology_large", "ner_drugs_greedy", "ner_jsl_enriched", "jsl_ner_wip_clinical", "ner_drugs_large", "ner_posology_greedy", "ner_jsl"]
@st.cache_resource
def clinical_pipeline(model_name):

  documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  sentenceDetector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")
      #.setExplodeSentences(True)

  tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")


  word_embeddings =  WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")


  pos_tagger = PerceptronModel()\
      .pretrained("pos_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token"])\
      .setOutputCol("pos_tags")


  dependency_parser = DependencyParserModel()\
      .pretrained("dependency_conllu", "en")\
      .setInputCols(["sentence", "pos_tags", "token"])\
      .setOutputCol("dependencies")


  #ADE
  ade_ner = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ade_ner")

  ade_ner_converter = NerConverterInternal() \
      .setInputCols(["sentence", "token", "ade_ner"]) \
      .setOutputCol("ade_ner_chunk")\
      .setWhiteList(["ADE"])
  
  #drug 
  clinical_ner = MedicalNerModel.pretrained(model_name, "en", "clinical/models") \
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("clinical_ner")

  if model_name == 'ner_jsl_enriched':

      clinical_ner_converter = NerConverterInternal()\
        .setInputCols(["sentence","token","clinical_ner"])\
        .setOutputCol("clinical_ner_chunk")\
        .setWhiteList(["Drug_BrandName","Drug_Ingredient"])\
        .setReplaceLabels({"Drug_Ingredient":"Drug_BrandName"})

      chunk_merger = ChunkMergeApproach()\
      .setInputCols("clinical_ner_chunk", "ade_ner_chunk")\
      .setOutputCol('final_ner_chunk')

  elif model_name == 'jsl_ner_wip_clinical':

      clinical_ner_converter = NerConverterInternal()\
        .setInputCols(["sentence","token","clinical_ner"])\
        .setOutputCol("clinical_ner_chunk")\
        .setWhiteList(["Drug_BrandName","Drug_Ingredient"])\
        .setReplaceLabels({"Drug_Ingredient":"Drug_BrandName"})

      chunk_merger = ChunkMergeApproach()\
      .setInputCols("clinical_ner_chunk", "ade_ner_chunk")\
      .setOutputCol('final_ner_chunk')
        
  elif model_name == 'ner_jsl':

      clinical_ner_converter = NerConverterInternal()\
        .setInputCols(["sentence","token","clinical_ner"])\
        .setOutputCol("clinical_ner_chunk")\
        .setWhiteList(["Drug_BrandName","Drug_Ingredient"])\
        .setReplaceLabels({"Drug_Ingredient":"Drug_BrandName"})

      chunk_merger = ChunkMergeApproach()\
      .setInputCols("clinical_ner_chunk", "ade_ner_chunk")\
      .setOutputCol('final_ner_chunk')

  else:
  
      clinical_ner_converter = NerConverterInternal()\
        .setInputCols(["sentence","token","clinical_ner"])\
        .setOutputCol("clinical_ner_chunk")\
        .setWhiteList(["DRUG"])\
        .setReplaceLabels({"DRUG": "Drug_BrandName"})
        
      chunk_merger = ChunkMergeApproach()\
      .setInputCols("clinical_ner_chunk", "ade_ner_chunk")\
      .setOutputCol('final_ner_chunk')
      
  
  reModel = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "final_ner_chunk", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(['drug-ade', 'ade-drug', 'drug_brandname-ade', 'ade-drug_brandname', 'Drug_BrandName-ADE', 'ADE-Drug_BrandName'])\
    .setRelationPairsCaseSensitive(False)\
    .setCustomLabels({"1": "is_related", "0": "not_related"})
  
  # convert chunks to doc to get sentence embeddings of them
  chunk2doc = Chunk2Doc()\
        .setInputCols("final_ner_chunk")\
        .setOutputCol("final_chunk_doc")

  sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
        .setInputCols(["final_chunk_doc"])\
        .setOutputCol("sbert_embeddings")\
        .setCaseSensitive(False)

  # filter ADE entity embeddings
  router_sentence_icd10 = Router() \
        .setInputCols("sbert_embeddings") \
        .setFilterFieldsElements(["ADE"]) \
        .setOutputCol("ade_embeddings")

  # filter DRUG entity embeddings
  router_sentence_rxnorm = Router() \
        .setInputCols("sbert_embeddings") \
        .setFilterFieldsElements(["Drug_Brandname"]) \
        .setOutputCol("drug_brandname_embeddings")

  # use ade_embeddings only
  icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm","en", "clinical/models") \
        .setInputCols(["ade_embeddings"]) \
        .setOutputCol("icd10cm_code")\
        .setDistanceFunction("EUCLIDEAN")
    
  # use drug_embeddings only
  rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm","en", "clinical/models")\
        .setInputCols(["drug_brandname_embeddings"]) \
        .setOutputCol("rxnorm_code")\
        .setDistanceFunction("EUCLIDEAN")


  ner_pipeline = Pipeline(stages=[
      documentAssembler,
      sentenceDetector,
      tokenizer,
      word_embeddings,
      pos_tagger,
      dependency_parser,
      ade_ner,
      ade_ner_converter,
      clinical_ner,
      clinical_ner_converter,
      chunk_merger,
      reModel,
      chunk2doc,
      sbiobert_embeddings,
      router_sentence_icd10,
      router_sentence_rxnorm,
      icd_resolver,
      rxnorm_resolver])

  empty_data = spark.createDataFrame([[""]]).toDF("text")
  model = ner_pipeline.fit(empty_data)
  return model

###################################################

st.header("Please select a sample clinical text to test selected NER model:")
sample_text = st.selectbox('Clinical Texts', [clinical_text1, clinical_text2, clinical_text3, clinical_text4, clinical_text5, clinical_text6, clinical_text7])
st.write(sample_text)
model_name = st.sidebar.selectbox("NER Model Name", model_name)



model_load_state = st.info(f"Loading pipeline '{model_name}'...")

model_load_state.empty()


def main():

  import pandas as pd
  import pyspark.sql.functions as F

  light_model = LightPipeline(clinical_pipeline(model_name))
  light_result = light_model.fullAnnotate(sample_text)


  chunks = []
  codes = []
  begin = []
  end = []
  entities = []

  for chunk, code in zip(light_result[0]['ade_ner_chunk'], light_result[0]['icd10cm_code']):
            
        begin.append(chunk.begin)
        end.append(chunk.end)
        chunks.append(chunk.result)
        entities.append(chunk.metadata['entity'])
        codes.append(code.result+'     [icd10cm_code]')

  df_1 = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 'entity':entities, 'code':codes})

  chunks = []
  codes = []
  begin = []
  end = []
  entities = []

  for chunk, code in zip(light_result[0]['clinical_ner_chunk'], light_result[0]['rxnorm_code']):
            
        begin.append(chunk.begin)
        end.append(chunk.end)
        chunks.append(chunk.result)
        entities.append(chunk.metadata['entity'])
        codes.append(code.result+'    [rxnorm_code]')

  df_2 = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 'entity':entities, 'code':codes})

  #Relationship between ADE and Drug_Brandname

  df_2['relations']=''
  for m, i in enumerate(df_2['chunks']):
    for k in  light_result[0]['relations']:
      if k.metadata['chunk1'] == i :
        df_2.loc[m, 'relations']=f"{df_2['relations'].loc[m]},{k.metadata['chunk2']}"
      elif k.metadata['chunk2'] == i:
        df_2.loc[m, 'relations']=f"{df_2['relations'].loc[m]},{k.metadata['chunk1']}"
  
  df_1['relations']=''
  for m, i in enumerate(df_1['chunks']):
   for k in  light_result[0]['relations']:
     if k.metadata['chunk1'] == i :
       df_1.loc[m, 'relations']=f"{df_1['relations'].loc[m]} {k.metadata['chunk2']}"
     elif k.metadata['chunk2'] == i:
       df_1.loc[m, 'relations']=f"{df_1['relations'].loc[m]} {k.metadata['chunk1']}"


  #Dataframe

  result_merge = pd.concat([df_1,df_2], ignore_index=True).fillna("")

  
  
  st.write("Number of Entities")
  
  data1 = df_1['entity'].value_counts()
  data2 = df_2['entity'].value_counts()

  #data = [(data1, data2)]
  #df_bar = pd.DataFrame(data, columns=['ADE', 'Drug_BrandName'])

  df_bar = pd.DataFrame({'ADE': data1, 'Drug_BrandName': data2})
  st.bar_chart(data=df_bar, width=0, height=0, use_container_width=True)

  st.write("Table")
  gb = GridOptionsBuilder.from_dataframe(result_merge)
  gb.configure_pagination()
  gridOptions = gb.build()
  AgGrid(result_merge, gridOptions=gridOptions)

  HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #CD5C5C; background-color: white; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; color:black">{}</div>"""

  visualizer_name = ["RelationExtractionVisualizer", "NerVisualizer"]
  visualizer_name = st.sidebar.selectbox("Select a Visualizer", visualizer_name)

  #def viz (data, col, doc):
    
  st.write("Visualizer")
  if visualizer_name == 'RelationExtractionVisualizer': 
    #NerVisualizer().primaryColor = "#32a852"
    #X=NerVisualizer().set_label_colors({'Drug_BrandName':'#a0daa9', 'ADE':'#e9897e'})
    vis = RelationExtractionVisualizer()
    #vis.display(light_results[0], 'relations', show_relations=True)
    raw_html = vis.display(light_result[0], 'relations', 'document', return_html=True)
    st.write(HTML_WRAPPER.format(raw_html), unsafe_allow_html=True)
  else:
    NerVisualizer().primaryColor = "#32a852"
    X=NerVisualizer().set_label_colors({'Drug_BrandName':'#a0daa9', 'ADE':'#e9897e'})
    raw_html = X.display(light_result[0], 'final_ner_chunk', 'document', return_html=True)
    st.write(HTML_WRAPPER.format(raw_html), unsafe_allow_html=True)

    

  #viz (light_result[0], 'relations', 'document')


if __name__ == '__main__':
    main()

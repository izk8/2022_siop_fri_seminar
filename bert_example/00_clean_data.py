import pandas as pd

# read in the data, shorten, and select
df = pd.read_json("/home/isaac/work/2022_SIOP_Fri_Seminar/data/indeed_com-indeed_com_usa_jobs__20200101_20200331_deduped_n_merged_20201027_043720103677161.ldjson", lines=True)
df = df[:10000] # shorten it for demo purposes
df = df[['job_title','category','job_description']]
df

# cat names, then figure out a number for each one, then add the number
category_names = df["category"].unique().tolist() 
all_cat_indicies = [category_names.index(label) for label in category_names]
df["cat_indicies"] = [category_names.index(label) for label in df["category"]]

labels_to_add = ['Sales','Restaurant-or-food-Service','Computer-or-internet', 'Upper-Management-or-consulting']
indices_to_add = [category_names.index(label) for label in labels_to_add]
y = [label if label in indices_to_add else -1 for label in df["cat_indicies"]]

# save data in dic 
job_dict = {}
job_dict["df"] = df
job_dict["category_names"] = category_names
job_dict["all_cat_indicies"] = all_cat_indicies
job_dict["labels_to_add"] = labels_to_add
job_dict["indices_to_add"] = indices_to_add
job_dict['y'] = y

job_dict.keys()

# save in python format
import pickle
f = open("job_dict.pkl","wb")
pickle.dump(job_dict,f)
f.close()

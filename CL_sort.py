# %%
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing.pool import ThreadPool as Pool
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from tempfile import TemporaryDirectory
from multiprocessing import cpu_count
from odf.opendocument import load
import matplotlib.pyplot as plt
from functools import partial
import tensorflow_hub as hub
from pathlib import Path
import tensorflow_text
from tqdm import tqdm
from PIL import Image
from odf import text
import pandas as pd
import numpy as np
import pytesseract
import argparse
import zipfile
import shutil
import docx
import fitz
import os

# %%
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("folder_path", help = "Folder path for program to start searching for files")
parser.add_argument("-l", "--lang", default = "", type = str, help = "Add anoter language for program to detect + english")
parser.add_argument("-m", "--max_page", default = 5, type = int, help = "Set max page count for reading pdf files")
parser.add_argument("-t", "--text_max", default= 12000, type = int, help = "Maximum amount of text that can be extract form a file")
parser.add_argument("-d", "--doc_img", action = "store_true", help="Extract data from image inside a doc file")
parser.add_argument("-p", "--show_path", action = "store_true", help="Show file's path while program running")
parser.add_argument("-i", "--image_size", default = 500, type = int, help = "Image size when extrcting image files")
parser.add_argument("-e", "--encoder", default = "", type = str, help = "change text encoder (tensorflow hub.load)")
parser.add_argument("-s", "--stage", default = 1, type = int, help = "set opration stage")

# %%
#Read Argument
cfg = vars(parser.parse_args())
f_path = cfg["folder_path"]         #req
lang = cfg["lang"]                  #l or lang
pdf_max_page = cfg["max_page"]      #m or max-page
img_doc = cfg["doc_img"]            #d or doc-img
show_path = cfg["show_path"]        #p or show-path
text_limit = cfg["text_max"]        #t or text-max
image_size = cfg["image_size"]      #i or image-size
stage = cfg["stage"]                #s or stage
print("loading encoder")
if cfg["encoder"] == "" :
  import tensorflow as tf
  embed = tf.saved_model.load("./Model/")
else :
  embed = hub.load(cfg["encoder"])

# %%
class extract :
    def __init__(self, lang = None, max_page = 5, img_doc = False, show_path = False, text_limit = 12000, image_size = 800) :
        self.max_page = max_page
        self.img_doc = img_doc
        self.show_path = show_path
        self.text_limit = text_limit
        self.img_size = image_size
        self.error_count = 0

        if lang == None : #set lang to eng if no language is input
            self.lang = "eng"
        elif lang == 'eng': #if it eng pass it
            self.lang = "eng"
        else : #if there is language input add eng after it
            self.lang = lang + "+" + "eng"
    
    def clean_up (self) :
        if os.path.exists("./extract") :
          shutil.rmtree("./extract")
        if os.path.exists("./encode") :
          shutil.rmtree("./encode")
        if os.path.exists("./temp") :
          shutil.rmtree("./temp")
            
    def find_files (self, path : str) -> list :
        types = ["pdf","docx", "doc", "odt", "txt", "png", "jpg", "jepg"]
        paths = []
        for f_type in types : #loop through types
            for data in list(Path(path).glob(f"**/*.{f_type}")) : #grab all file with the same suffix
                paths.append(str(data)) #add it to paths
        return paths

    def path_pos (self, path : list, index : int) -> list :
        out = []
        pos = index * -1
        for data in path :
          data = os.path.dirname(str(data).replace("\\", "/"))
          out.append(data.split("/")[pos])
        return out

    def remove_things (self, text : str) -> str :
        replace_list = ["\n", "\t", "!", '"', "'", ' ', '#', '“', '”', '-', '…']
        for data in replace_list : #remove things from input text
            text = text.replace(data, '')
        return text
    
    def read_docx (self, path : str) -> str :
        file = docx.Document(path) #open doc file
        limit_reach = False
        pre_out = ''
        out = ''
        for paragraph in file.paragraphs : # extract text from doc by paragraphs
            paragraph = self.remove_things(paragraph.text) #remove unwanted charector from paragraph
            if paragraph != '' : #if paragraph is not empty
                pre_out = pre_out + paragraph
            if len(pre_out) > self.text_limit : #check if the limit is reached
                limit_reach = True
                break
            else:
                out = pre_out
        if self.img_doc == True and limit_reach == False : #check if user wants to extract data from img 
            with TemporaryDirectory() as tempdir : #create temp dir
                file = zipfile.ZipFile(path).extractall(tempdir) #extract doc data to tempdir
                file = Path(str(tempdir) + "/word/media") #go to img dir
                for pic in self.find_files(file) : #find img and read them
                    out = out + self.read_img(pic)
            
        return out

    def read_odt (self, path : str) -> str :
        file = load(path) #read odt
        limit_reach = False
        pre_out = ''
        out = ''
                
        for paragraph in file.getElementsByType(text.P) : #extract text by paragraphs
            for data in str(paragraph) : #remove unwanted charector from paragraph
                data = self.remove_things(data)
                if data != '' : #if paragraph is not emty
                    pre_out = pre_out + data
                if len(pre_out) > self.text_limit : #check if the limit is reached
                    limit_reach = True
                    break
                else:
                    out = pre_out
                            
        if self.img_doc == True and limit_reach == False: #check if user wants to extract data from img 
            with TemporaryDirectory() as tempdir : #create temp dir
                file = zipfile.ZipFile(path).extractall(tempdir) #extract doc data to tempdir
                file = Path(str(tempdir) + "/Pictures") #go to img dir
                for pic in self.find_files(file) : #find img and read them
                    out = out + self.read_img(pic)
        return out
    
    def read_pdf (self, path : str) -> str :
        with TemporaryDirectory() as tempdir : #create temp dir
            file_pos = []
            pre_out = ''
            out = ''
            doc = fitz.open(path) #open pdf
                
            for count,page in enumerate(doc): #open each page of pdf
                if count == self.max_page : #break if it reaches max_page
                    break
                else :
                    file_pos.append(f"{tempdir}\page_{count}.png") #add files pos
                    pix = page.get_pixmap() #convert that page to picture
                    pix.save(f"{tempdir}\page_{count}.png") #save it as png
    
            for image_file in file_pos: #loop through file pos
                text = self.read_img(image_file,True) #extract text from pic (also pass through language to tesseract)
                pre_out = pre_out + text
                if len(pre_out) > self.text_limit : #check if text limit is reached
                    break
                else:
                    out = pre_out
                out = out + text
            return out
    
    def read_text (self, path : str) -> str :
        with open(path, "r", encoding="UTF-8", errors="ignore") as file : #open file using UTF-8 and ignore errors
            temp = file.readlines() #extract data by lines
            pre_out = ''
            out = ''
            for data in temp : #loop though each line
                data = self.remove_things(data)
                if data != '' : #if line is not emty
                    pre_out = pre_out + data
                if len(pre_out) > self.text_limit : #check if the text limit is reached
                    break
                else:
                    out = pre_out
        return out
    
    def read_img (self, path : str, bypass = False) -> str :
        data = Image.open(path)
        if bypass == True :
          ratio = data.width / data.height
          resized = data.resize((int(self.img_size*ratio),self.img_size))
          return self.remove_things(str(pytesseract.image_to_string(resized, lang=self.lang))) #return data that is read from img
        else :
          return self.remove_things(str(pytesseract.image_to_string(data, lang=self.lang))) #return data that is read from img
    
    def read_files (self, path : str) -> list:
        #Call function to extract the file and pass some paramiters
        def select_func (path : str, suffix : str) :
            res = ''
            if suffix == ".pdf" :
                res = self.read_pdf(path)

            elif suffix == ".docx" or suffix == ".doc":
                res = self.read_docx(path)

            elif suffix == ".odt" :
                res = self.read_odt(path)

            elif suffix == ".txt" or suffix == ".TXT":
                res = self.read_text(path)

            elif suffix == ".png" or suffix == ".jpg" or suffix == ".jpeg" :
                res = self.read_img(path)

            return res

        #Acual function start here
        file = Path(path)
        try : res = (select_func(file,file.suffix)) #call select_func and passthrough the file suffix (.doc .pdf .jpg)
        except : res = "error"
        if self.show_path == True : #print the file's path if show_path is enabled
            print(file)

        if not os.path.exists("./extract") :
            os.mkdir("./extract")

        if res != "error" :
            temp_file = (str(file).replace("\\", "/")).split("/")[-1].replace(file.suffix, "_extract.txt")
            file = Path(f"./extract/{temp_file}")
            if os.path.exists(file) :
                counter = 1
                while os.path.exists(file) :
                    file = Path(f"./extract/{counter}_{temp_file}")
                    counter += 1
            with open(file, "w", encoding="UTF-8", errors="ignore") as f :
                f.write(path + "\n")
                f.write(res)
        else : 
            file = "error"
            self.error_count += 1
        return file

    def main (self, f_path, d_type) -> list :
        if type(f_path) == str :
            files = self.find_files(f_path) #get every file path that program supports
        else :
            files = f_path #bypass find files function if input is list
            
        p =  Pool(processes=cpu_count()) #create pool with process node = cpu cores

        if self.show_path == False : #if show path is disabled show progressbar
            res = tqdm(p.imap(partial(self.read_files),files), total=len(files), desc=f"Extracting text for {d_type}")
        else : #if show path is enabled show file's path
            res = p.imap(partial(self.read_files),files)
        p.close()

        out = []
        for data in res :
            if data != "error" :
                out.append(data)
        print(f"Bad Files : {self.error_count}")
        return out

# %%
#Argument Edition + Final
df_label = None
df_data = None
label_encode = None
data_encode = None
move_df = None
if stage == 1 :
  file_ext = extract(lang, pdf_max_page, img_doc, show_path, text_limit,image_size)
  file_ext.clean_up()  
  main_path = file_ext.find_files(f_path)
  sub = file_ext.path_pos(main_path,1)
  root = (str(f_path).replace("\\", "/")).split("/")[-1]
  remove_l = []
  data = []
  data_sub = []
  for index, path in enumerate(sub) :
    if path == root :
     remove_l.append(index)
  for i in remove_l :
    data.append(main_path[i])
    data_sub.append(sub[i])
  remove_l = remove_l[::-1]
  for i in remove_l :
    sub.pop(i)
    main_path.pop(i)
  df_label = pd.DataFrame({"Path" : main_path, "sub_dir" : sub})
  df_data = pd.DataFrame({"Path" : data, "sub_dir" : data_sub})
  print("stage 1")
  if not os.path.exists("./temp") :
    os.mkdir("./temp")
  if len(df_label["Path"]) == 0 or len(df_data["Path"]) == 0:
    print("No sample/label files switching to clustering mode")
    print("WARNING DO NOT USE THIS MODE WITH A LOT OF DATA")
    df_data = pd.concat([df_label, df_data])
    df_data = pd.DataFrame({"Path" : file_ext.main(df_data["Path"], "data")})
    df_data.to_csv("./temp/extract_data.csv", index=False)
    stage = 2.2
  else :
    df_label = pd.DataFrame({"Path" : file_ext.main(df_label["Path"], "label")})
    df_data = pd.DataFrame({"Path" : file_ext.main(df_data["Path"], "data")})
    df_label.to_csv("./temp/extract_label.csv", index=False)
    df_data.to_csv("./temp/extract_data.csv", index=False)
    stage = 2.1

# %%
def encode (df : pd.DataFrame, P_name, size) :
  folder_path = f"./encode/{P_name}/"

  if not os.path.exists("./encode") :
    os.mkdir("./encode")
  os.mkdir(folder_path)
  old_path = []
  paths = []

  for data in tqdm(df["Path"], desc=f"Encoding {P_name}") :
    data = Path(("./" + str(data)).replace("\\","/"))
    with open(data, "r", encoding="UTF-8", errors="ignore") as f :
      f_data = f.read().split("\n")
    
    f_path = f_data[0].replace("\\", "/")
    f_path = f_path.split("/")[-1]
    f_path = f_path.replace(Path(f_path).suffix, "_encode.txt")
    f_path = Path(folder_path + f_path)
    old_path.append(f_data[0])
    paths.append(f_path) 
    if size == 1 :
      vector = embed(f_data[1])[0].numpy()
    else :
      vector = embed(f_data[1]).numpy()
    vector.dump(f_path)
  return pd.DataFrame({"Old_path" : old_path, "Path" : paths})

# %%
if stage == 2.1 :
  if os.path.exists("./encode/label") :
    shutil.rmtree("./encode/label")
  if type(df_label) == type(None) :
    df_label = pd.read_csv("./temp/extract_label.csv")
  print("stage 2.1")
  label_encode = encode(df_label, "label", 2)
  label_encode.to_csv("./temp/encode_label.csv", index=False)
  stage = 3.1

if stage == 2.2 :
  if os.path.exists("./encode/data") :
    shutil.rmtree("./encode/data")
  if type(df_data) == type(None) :
    df_data = pd.read_csv("./temp/extract_data.csv")
  print("stage 2.2")
  data_encode = encode(df_data, "data", 1)
  data_encode.to_csv("./temp/encode_data.csv", index=False)
  stage = 3.2

if stage == 3.1 :
  if os.path.exists("./encode/data") :
    shutil.rmtree("./encode/data")
  if type(df_data) == type(None) :
    df_data = pd.read_csv("./temp/extract_data.csv")
  print("stage 3.1")
  data_encode = encode(df_data, "data", 2)
  data_encode.to_csv("./temp/encode_data.csv", index=False)
  stage = 4.1

# %%
#cosine
if stage == 4.1 :
  if type(data_encode) == type(None) :
    data_encode = pd.read_csv("./temp/encode_data.csv")
  if type(label_encode) == type(None) :
    label_encode = pd.read_csv("./temp/encode_label.csv")
  print("stage 4.1")
  Move_to = []
  File_path = []
  for counter1, path_data in enumerate(tqdm(data_encode["Path"], desc="Processing Data")) : #get encode data path (data)
    best = 0 #reset 
    best_list = []
    best_list_count = []
    data = np.load(path_data, allow_pickle=True) #load encoded data

    for counter2, path_label in enumerate(label_encode["Path"]) : #get encode data (label)
      label = np.load(path_label, allow_pickle=True) #load encoded data
      co = cosine_similarity(data, label) #cosine similarity
      if co >= best : #if it beat best score save it index and socre
        best = co
        best_count = counter2
    Move_to.append(os.path.dirname(label_encode["Old_path"][best_count].replace("\\", "/"))) #add real file path (to be moved to)
    File_path.append(data_encode["Old_path"][counter1].replace("\\", "/")) #add real file path (data wanted to be sort)
    move_df = pd.DataFrame({"Predict" : Move_to, "File_path" : File_path}) #make dataframe from them
    move_df.to_csv("./temp/move_df.csv", index=False)
  stage = 5.1

# %%
#find optimal cluster count (k)
if stage == 3.2 :
  if type(data_encode) == type(None) :
    data_encode = pd.read_csv("./temp/encode_data.csv")
  print("stage 3.2")
  print("clustering")
  sil = []
  vector = []
  kmax = round(data_encode.shape[0]*0.2)
  for data in data_encode["Path"] : #this will cause memory to go kaboom
    vector.append(np.load(data, allow_pickle=True))
  vector  = np.array(vector)
  for k in range(2, kmax+1):
    model = SpectralClustering(n_clusters=k, random_state = 69420)
    model.fit_predict(vector)
    labels = model.labels_
    sil.append(silhouette_score(vector, labels, metric = 'euclidean'))
  #create data cluster using optimak k
  k_end = np.array(sil).argmax()
  model = SpectralClustering(n_clusters= k_end, random_state=69420)
  model.fit(vector)
  data_encode['cluster'] = model.labels_
  data_encode.to_csv("./temp/encode_data.csv", index=False)

# %%
#Move files into their location based on cluster
if stage == 3.2 :
  if type(data_encode) == type(None) :
    data_encode = pd.read_csv("./temp/encode_data.csv")
  print("moving files")
  for pain in np.sort(data_encode["cluster"].unique()) :
      root = Path(f"{f_path}/cluster {pain}/")
      if not os.path.exists(root) :
          root.mkdir()
      for path in data_encode[data_encode['cluster'] == pain]['Old_path'] :
          path = Path(path)
          dest = Path(str(root) + "/" + str(path.name))
          path.rename(dest)
if stage == 5.1 :
  if type(move_df) == type(None) :
    move_df = pd.read_csv("./temp/move_df.csv")
  print("stage 5.1")
  print("moving files")
  for index, data in enumerate(move_df["Predict"]) :
      path = Path(move_df["File_path"][index])
      dest = Path(data + "/" + path.name)
      path.rename(dest)
print("Complete.")



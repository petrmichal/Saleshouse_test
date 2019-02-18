
# coding: utf-8

# In[ ]:


### installs
def run(env, jsn = 'missing'):
  #!pip install google-cloud-automl
  #!pip install google-cloud-storage
  #!pip install opencv-python==3.3.0.9
  #!pip install keras
  #!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
  #!pip install tensorflow==1.12.0


# In[34]:


### imports
  
  from google.cloud import automl_v1beta1 # Imports the Google Cloud client library
  from google.cloud.automl_v1beta1.proto import service_pb2
  #from google.datalab import storage as strg
  from google.cloud import storage
  from IPython.display import Image
  from PIL import Image as Img
  from PIL.ExifTags import TAGS
  from imageai.Detection import ObjectDetection
  from google.cloud.storage import Blob
  import datetime
  import google.cloud.bigquery as bq
  import tensorflow as tf
  import matplotlib.pyplot as plt
  import math
  import cv2
  import numpy as np
  import pandas as pd
  import glob
  import json
  import io
  from keras import backend as K
  from pytz import timezone  
  from  modules.logger import writelog
  
  from skimage import filters

  
  import gc
  import os
  from skimage.color import rgb2lab, deltaE_cie76   ### alternative vertical shelves detection  
  import copy
  import pickle  
  import re  
    
# In[ ]:

# env variables
  output_bucket_name = '{}-shelfinspector-outputs'.format(env)
  bq_dataset = '{}_db'.format(env)

# Instantiates a client
  storage_client = storage.Client()

  # The name for the new bucket
  bucket_name = 'saleshousephotos'

  bucket = storage_client.get_bucket(bucket_name)
  blobs = bucket.list_blobs()

  ######### toto tu nebude ##########
  photos = []
  for blob in blobs:
    if 'photos' in blob.name:
      photos.append(blob.name)
  photos = photos[1:]
  ###################################

  ########## michal.mlaka ###########
  # photos = URL k dane fotce ktera prijde
  # pripluje photo_ID
  if jsn == 'missing':
    bucket_name = 'saleshouse-test-output'
    bucket = storage_client.get_bucket(bucket_name)
    jsn = Blob("newphoto.json", bucket)
    jsn = json.loads(jsn.download_as_string().decode('utf-8'))
    modelpath = './'
    outputpath = modelpath
 
  photos = jsn['name']
  bucket_name = jsn['bucket'] 
  bucket = storage_client.get_bucket(bucket_name)  
  recognition_start = datetime.datetime.now(timezone('Europe/Berlin')).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
  photo_load = pd.to_datetime(jsn['timeCreated']).tz_localize('UTC').tz_convert('Europe/Berlin').strftime("%Y-%m-%dT%H:%M:%S.%fZ")
  
  if 'load_type' in jsn:
      load_type = jsn['load_type']
  else:
      load_type = 'default'
 
  modelpath = '/shelf-inspector/models/'
  outputpath = '/shelf-inspector/tmp-outputs/'

 #jsn = {"bucket": "saleshouse-test-pipeline", "contentType": "image/jpeg", "crc32c": "uH7RjA==", "etag": "CN/0h77T5d4CEAE=", "generation": "1542809022757471", "id": "saleshouse-test-pipeline/photos/15428090045761973780038.jpg/1542809022757471", "kind": "storage#object", "md5Hash": "S+F908zB4FNUDAoqL07QOA==", "mediaLink": "https://www.googleapis.com/download/storage/v1/b/saleshouse-test-pipeline/o/photos%2F15428090045761973780038.jpg?generation=1542809022757471&alt=media", "metageneration": "1", "name": "photos/15428090045761973780038.jpg", "selfLink": "https://www.googleapis.com/storage/v1/b/saleshouse-test-pipeline/o/photos%2F15428090045761973780038.jpg", "size": "2611476", "storageClass": "REGIONAL", "timeCreated": "2018-11-21T14:03:42.757Z", "timeStorageClassUpdated": "2018-11-21T14:03:42.757Z", "updated": "2018-11-21T14:03:42.757Z"}
  #bucket_name = jsn['bucket']


# In[3]:


### functions neeeded

  def blur_image(image, amount=5):
      '''Blurs the image
      Does not affect the original image'''
      kernel = np.ones((amount, amount), np.float32) / (amount**2)
      return cv2.filter2D(image, -1, kernel)

  def colour_frame(img, frame, width=5, colour=[0, 0, 0]):
      """"""
      x1, y1, x2, y2 = frame['box_points']
      img[y1:y2, x1-width:x1+width] = colour
      img[y1:y2, x2-width:x2+width] = colour
      img[y1-width:y1+width, x1:x2] = colour
      img[y2-width:y2+width, x1:x2] = colour
      return img


  def colour_map(shelf):
      MAP = {0: [255, 0, 0], 1: [0, 0, 255], 2: [0, 255, 0], 3: [0, 255, 255], 4: [255, 255, 0], 5: [24, 49, 216], 6: [0, 209, 229], 7: [46, 60, 62], 8: [255, 128, 0], 9: [255, 0, 191]}
      if shelf < 0:
          return [0, 0, 0 ]
      return MAP.get(shelf % len(MAP))

    
  def compute_dominance_relations(detections):
      """"""
      for i, frame in enumerate(detections):
          x1, y1, x2, y2 = frame['box_points']
          for j, frame_ in enumerate(detections):
              x1_, y1_, x2_, y2_ = frame_['box_points']
              if y2 < y1_:
                  dominated_by = frame_.setdefault('dominated_by', set())
                  dominated_by.add(i)


  def compute_seed_shelves(detections, polish=1/4):
      """"""
      frame_indeces = {i for i in range(len(detections))}    
      shelf = -1
      shelves = []
      while frame_indeces:
          shelf_frames = set()  
          for i in frame_indeces:
              frame = detections[i]
  #            current_frame = colour_frame(main_obj, frame)
  #            Image.fromarray(current_frame, 'RGB').show()
              dominated_by = (detections[i] for i in frame.get('dominated_by', {}))
              if all(dominating_frame.get('shelf', float('inf')) <= shelf 
                     for dominating_frame in dominated_by):
                  x1, y1, x2, y2 = frame['box_points']
                  upper_shelf_bottoms = [detections[i]['box_points'][3] for i in shelves[-1]] if shelf > -1 else []
                  if sum(y2_ >= y2 - GAP for y2_ in upper_shelf_bottoms) > polish*len(upper_shelf_bottoms):
  #                if all(y2_ >= y2 - GAP for y2_ in upper_shelf_bottoms):

                      frame['shelf'] = shelf
                      shelves[-1].add(i)
                  else:
                      frame['shelf'] = shelf + 1
                      next_shelf = True
                      shelf_frames.add(i)
  #                main_obj = colour_frame(main_obj, frame, colour=colour_map(shelf))
  #                img = Image.fromarray(main_obj, 'RGB')
          frame_indeces -= shelf_frames
          if shelf > -1:
              frame_indeces -= shelves[-1]
          shelves.append(shelf_frames)

          shelf += 1 if next_shelf else 0
      return detections


  def detect_nonshelves(detections, main_obj):
      """"""
      shelves = [set()  for _ in range(20)]
      for iframe, frame in enumerate(detections):
          shelves[frame['shelf']].add(iframe)

      image_width = main_obj.shape[1]
      to_remove = []
      for ishelf, shelf in enumerate(shelves[:-1]):
          lower_tops = [(detections[j]['box_points'][1], detections[j]['box_points'][2]) for j in shelves[ishelf+1]]
          if not lower_tops:
              continue
          noshelf = []
          for i in shelf:
              frame = detections[i]
              frame_bottom = frame['box_points'][3]
              frame_top = frame['box_points'][1]
              if sum(y1 < frame_bottom for y1, y2 in lower_tops) > 0*len(lower_tops)/2:
  #                frame['shelf'] = -1
                  noshelf.append(frame)
  #        Image.fromarray(colour_shelves(noshelf, main_obj.copy()), 'RGB').show()
          safety_condition = noshelf and (min(f['box_points'][0] for f in noshelf) > image_width*1/2
                          or max(f['box_points'][2] for f in noshelf) < image_width*1/2)
          if safety_condition:
              for f in noshelf:
                  f['shelf'] = -1
              to_remove.extend(noshelf)
      for frame in detections:
          if to_remove:
              if frame['box_points'][2] > image_width/2:
                  if frame['box_points'][2] >= min(f['box_points'][0] for f in to_remove) > image_width/2:
                      frame['shelf'] = -1
              elif frame['box_points'][2] < image_width/2:
                  if frame['box_points'][0] <= max(f['box_points'][2] for f in to_remove) < image_width/2:
                      frame['shelf'] = -1
      return detections


  def detect_shelves(detections, polish=1/4):
      """Detekuje regaly na zaklade detekci lahvi. Vadi FALSE POSITIVES.
      V prvni iteraci je ale treba ponechat vysi FALSE POSITVES, abychom meli nizsi FALSE NEGATIVES,
      nebot FALSE NEGATIVES zase budou vadit nasledujicimu `correct` kroku.
      V pripade ze se v dalsich iteracich snizi FALSE POSITIVES, bude mozna treba zvysovat parametr `polish` smerem k 0.99.
      Parametr polish sleva "falesne regaly" do jednoho.
      Kazdy dict v `detections` dostane novy atribut `shelf` 
      """
      compute_dominance_relations(detections)   
      detections = compute_seed_shelves(detections, polish)
      return detections

  def correct_shelves(detections, main_obj):
      """Odstrani lahve stojici v sousednich regalech. Vadi FALSE NEGATIVES. 
      Prvky v `detections` k odstraneni dostanou `shelf=-1`
      Input
      ----
      detections : list of dicts
          detections with `shelf` attribute from `detect` method
      main_obj : array 
          image from the bottle detector
      """
      detections = detect_nonshelves(detections, main_obj)
      return detections

  def colour_shelves(detections, main_obj):
      """"""
      for frame in detections:
          if 'shelf' in frame:
              main_obj = colour_frame(main_obj, frame, colour=colour_map(frame['shelf']))
      return main_obj

  def colour_shelves_prediction(detections, main_obj):
      """"""
      for i, frame in enumerate(detections):
        if 'class' in frame:
            if frame['class'] == 'rb':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(4))
            elif frame['class'] == 'rbsf':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(6))
            elif frame['class'] == 'rbred':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(2))
            elif frame['class'] == 'rbblue':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(1))			 
            elif frame['class'] == 'bigshock':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(0))			 
            elif frame['class'] == 'rbturned':
              main_obj = colour_frame(main_obj, frame, colour=colour_map(4))	            #else: 
              #main_obj = colour_frame(main_obj, frame, colour=colour_map(0))
      return main_obj


  def rotateImage(image, angle):
    img = Img.fromarray(image, 'RGB')
    img = img.rotate(angle, expand=True)
    img = np.array(img)
    return img


  def detectVertShelf(image, color_threshold=10, minLineLength=300):
    """Detekuje svisle hrany (regaly) na zaklade metriky na RGB.
        Porovna "vzdalenost" barvy regalu od barev v regalu, vzdalenym pixelum da cernou barvu,
        pak se v takovem obrazku detekuji hrany  

      image ... vstupni obrazek, jako numpy array
      color_threshold ... threshold vzdalenosti barvy od barvy regalu
      minLineLength ... minimalni deka hrany pro detekci
      
      vraci puvodni obrazek a detekovane lines (viz cv2.HoughLinesP doc)
    """

    lab = rgb2lab(image)
    regal = [225, 220, 180]   ### approx. color of shelf (globus sediva)
    
    regal_3d = np.uint8(np.asarray([[regal]]))
    dE_regal= deltaE_cie76(rgb2lab(regal_3d), lab)
    
    image_res = image.copy()
    image_res[dE_regal >= color_threshold] = [0,0,0]   ## far away from shelf color -> black
    gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
    #minLineLength=100
    lines = cv2.HoughLinesP(image=gray,rho=1,theta=np.pi, threshold=20,lines=np.array([]), minLineLength=minLineLength, maxLineGap=3)
    
    return image, lines
  
  
  def shiftDetections(detec, shift):
    ### shift the boxes about the difference of img_chopped and unchopped img
   
    for i in range(len(detec)):
      #print(i)
      detec[i]['box_points'][0] = shift + detec[i]['box_points'][0]
      detec[i]['box_points'][2] = shift + detec[i]['box_points'][2]
    
    return detec  


  def planogram_check(obj, planograms, row, column, store_location, date, start_pix=None, end_pix=None):
    """
    The function takes as input the information from both photo product
    recognition reality and the target planogram. It outputs a DataFrame
    containing aggregated information about their similarity.
    Parameters
    ----------
    obj : Dictionary
        Contains information about pixel allocations to different products
        in a given row and column. Example input format:
            obj = {"label": ["rblue", ...], "location" : [[(0,0), (121, 134),...],[...],...]}
    planograms : Dictionary
        Contains parsed planograms. It is an output from read_all_planograms().
    row : Integer
        Row location of the obj in the whole shelf.
    column : Integer
        Row location of the obj in the whole shelf.
    store_location : String
        Location of the store from the target planogram file name. Example
        input: "Zlicin"
    start_pix : Integer
        Starting (leftmost) pixel of the given cell.
    end_pix : Integer
        Last (rightmost) pixel of the given cell.
    Returns
    -------
    plan_check_df : DataFrame
        DataFrame containing aggregated information about similarity between
        planogram and photo reality
    """

# =============================================================================
# load the target planogram part
# =============================================================================
    # find the correct index in the planograms dictionary
    plan_dates = list(map(lambda x:x.split("_")[0],planograms["plan_name"]))
    plan_stores = list(map(lambda x:x.split("_")[2],planograms["plan_name"]))
    plan_names_df = pd.DataFrame({"plan_dates" : plan_dates, "plan_stores": plan_stores})
    target_ind = plan_names_df.index[(plan_names_df["plan_dates"] == date) & (plan_names_df["plan_stores"] == store_location)].tolist()[0]
    # get the target planogram into df
    plan = planograms["plan_df"][target_ind]
    # zoom into the target planogram part
    plan = plan.loc[(plan['row'] == row) & (plan['column'] == column)]

# =============================================================================
# transform photo reality into checking matrix
# =============================================================================
    labels = obj["label"]
    num_labels = len(labels)
    # starting and ending pixels for measuring row size
    pixels = [item for sublist in obj["location"] for item in sublist]
    pixels = np.array(sum(pixels, ()))
    start_pix = min(pixels[np.nonzero(pixels)]) if start_pix is None else start_pix # minimal nonzero
    end_pix = max(pixels) if end_pix is None else end_pix
    row_size = end_pix - start_pix + 1

    # checking matrix dimensions: 0 - label, 1 - pixels
    real_check_mat = np.full((num_labels, row_size), 0 ,dtype=float)

    # fill the checking matrix in a loop over products
    for i in range(num_labels):
        # pixels containing the target label product
        label_pixels = obj["location"][i][1:] # remove (0,0)
        label_pixels = np.array(sum(label_pixels, ())) - start_pix
        # write into the table, loop trough starting pixel starting points
        for p in range(int(len(label_pixels)/2)):
            real_check_mat[i, label_pixels[2 * p] : label_pixels[2 * p + 1] + 1] = 1

# =============================================================================
# transform planogram into checking matrix
# =============================================================================
    # checking matrix dimensions: 0 - label, 1 - pixels
    plan_check_mat = np.full((num_labels, row_size), 0 ,dtype=float)

    # fill the checking matrix in a loop over products
    for i in range(num_labels):
        # filter down to the given label
        plan_label = plan.loc[plan['label'] == labels[i]]
        plan_label = plan_label.reset_index().drop(columns="index")
        # loop trough individual blocks and write into the table
        for b in range(plan_label.shape[0]):
            # translate cell share start and end points to pixels
            cell_share_start_pix = int(np.ceil(plan_label.loc[b, "cell_share_start"] * row_size))
            cell_share_end_pix = int(np.ceil(plan_label.loc[b, "cell_share_end"] * row_size))
            plan_check_mat[i, cell_share_start_pix : cell_share_end_pix] = 1

# =============================================================================
# Checking similarity between photo reality and planogram
# =============================================================================
    # share of cell where the products was planned but is missing
    missing_share = np.mean((real_check_mat < plan_check_mat)*1, axis=1)

    # share of cell where the product was not planned but is there
    extra_share = np.mean((real_check_mat > plan_check_mat)*1, axis=1)

    # originally planned
    planogram_share = np.mean((plan_check_mat)*1, axis=1)

    # combining into the resulting output
    plan_check_df = pd.DataFrame({"label": labels,
                                  "planogram_share": planogram_share,
                                  "missing_share": missing_share,
                                  "extra_share": extra_share})

    return plan_check_df

  def parse_store_location(store_string):
    """
    The function converts store string from the photo loading web application
    and transforms it into the correct store_location input for the
    planogram_check() function.
    Parameters
    ----------
    store_string : String
        String from web application containing store identification code
        in the round brackets. Example format:
            "Poděbradská 293 Pardubice Pardubice VII 53009 (S2318CZ)"
    Returns
    -------
    store_location : String
        String contatining unique store location name. It serves as an output
        to the planogram_check() function.
    """

# =============================================================================
# Create DataFrame of store codes and names
# =============================================================================
    store_names = ["Chomutov", "Ceske Budejovice", "Brno",
                   "Cerny Most", "Plzen", "Cakovice",
                   "Karlovy Vary", "Ostrava", "Pardubice",
                   "Olomouc", "Zlicin", "Liberec",
                   "Opava", "Havirov", "Usti"]

    store_codes = ["S2325CZ", "S2321CZ", "S2320CZ",
                   "S2323CZ", "S2324CZ", "S2315CZ",
                   "S2316CZ", "S2329CZ", "S2318CZ",
                   "S2322CZ", "S2753CZ", "S2317CZ",
                   "S2328CZ", "S2995CZ", "S2319CZ"]
    # create the code df
    df_store_codes = pd.DataFrame({"store_codes": store_codes,
                                   "store_names": store_names})


# =============================================================================
# Extract the code and transform it into the store names
# =============================================================================

    # extract store identification code
    id_code = re.search('\((.*)\)', store_string).group(1)

    # find the appropriate name for the code in the store_codes df
    store_location = df_store_codes.loc[df_store_codes["store_codes"] == id_code, "store_names"].values[0]

    return store_location


# In[4]:
  def filterDetections(sizes_x_arr, sizes_y_arr, detections_all, extracted_all, is_big):

    val_x = filters.threshold_otsu(sizes_x_arr)
    val_y = filters.threshold_otsu(sizes_y_arr)
  
    over_x = len(sizes_x_arr[sizes_x_arr>val_x])
    under_x = len(sizes_x_arr[sizes_x_arr<val_x])
    over_y = len(sizes_y_arr[sizes_y_arr>val_y])
    under_y = len(sizes_y_arr[sizes_y_arr<val_y])
  
    if is_big:
        if over_x < under_x:
            val_x = 10000      
    if True:    
        if (over_x < under_x) and (over_y < under_y) :        
                detections_wo = list()
                extracted_wo = list()
                for idx, l in enumerate(detections_all):
                  if ((l['box_points'][2] - l['box_points'][0]) < val_x)  and ((l['box_points'][3] - l['box_points'][1]) < val_y) :
                      detections_wo.append(l)  
                      extracted_wo.append(extracted_all[idx])

        elif (over_x < under_x) and (over_y > under_y) :
                detections_wo = list()
                extracted_wo = list()
                for idx, l in enumerate(detections_all):
                  if ((l['box_points'][2] - l['box_points'][0]) < val_x)  and ((l['box_points'][3] - l['box_points'][1]) > val_y) :
                      detections_wo.append(l)
                      extracted_wo.append(extracted_all[idx])
        elif (over_x > under_x) and (over_y < under_y) :               
                detections_wo = list()
                extracted_wo = list()
                for idx, l in enumerate(detections_all):
                  if ((l['box_points'][2] - l['box_points'][0]) > val_x)  and ((l['box_points'][3] - l['box_points'][1]) < val_y) :
                      detections_wo.append(l)
                      extracted_wo.append(extracted_all[idx])
        else:
                detections_wo = list()
                extracted_wo = list()
                for idx, l in enumerate(detections_all):
                  if ((l['box_points'][2] - l['box_points'][0]) > val_x)  and ((l['box_points'][3] - l['box_points'][1]) > val_y) :
                      detections_wo.append(l)
                      extracted_wo.append(extracted_all[idx])
    
    return detections_wo, extracted_wo    


  def isInShelf(box, upperLine, lowerLine):
    """  Check if the box is between those lines, upper < lower (0,0, is left-upper corner)  
    """

    p = box['box_points']
    box_height = p[3] - p[1]
    box_height
    cut_box_height = min(lowerLine, p[3]) - max(upperLine, p[1]) 
    if box_height - cut_box_height > 1/4*box_height:  ## how much I cut, too much cut -> not in this shelf
      return False
    else: 
      return True
  
  
  def getShelves(detections, lines):
    """ assign shelves to detections
    """

    for idx,det in enumerate(detections):
      dist_to_shelf = np.zeros(len(lines))
      b_points = det['box_points']
      b_height = b_points[3] - b_points[1] 
      for l in range(len(lines)): 
        dist_to_shelf[l] = lines[l] - b_points[1]  ### distance of upper-left corner from lines
        if dist_to_shelf[l] < 0:
            dist_to_shelf[l] = 100000 ## sth huge  
      #print(dist_to_shelf)             
      det['shelf'] = np.argmin(dist_to_shelf)
    return detections  

  def order_shelf_map(order):
      """ get first row in the photo and column (according to planograms) from order of photo (according to schema they took photos) 
      """
      MAP = {0: [0, 0], 1: [3, 0], 2: [0,1], 3: [3, 1], 4: [0, 2], 5: [3, 2]}
      if order < 0:
          return [0, 0]
      return MAP.get(order % len(MAP))


### vyhodit???
 # from __future__ import absolute_import
 # from __future__ import division
 # from __future__ import print_function

  ### functions for local model
  def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph


  def read_tensor_from_image_file(file_name,
                                  input_height=299,
                                  input_width=299,
                                  input_mean=0,
                                  input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
          file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
          tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
      image_reader = tf.image.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


  def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

###. dummy comment
  def local_recognizer(file_name, graph):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"

    #graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
          input_height=input_height,
          input_width=input_width,
          input_mean=input_mean,
          input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    l = top_k[0]

    return labels[l], results[l]


# In[7]:

  print('OK fn')   
### detection of objects for chopping the photo

  #for i in range(len(photos)):
  for i in range(1,2):
    tmp = bucket.blob(photos).download_as_string()
    img = Img.open(io.BytesIO(tmp))

    info = img._getexif()
    for tag, value in info.items():
        key = TAGS.get(tag)
        if key == 'Orientation':
            print(key + ': ' + str(value))
            orientation=value 
    
    img = np.array(img)

    which_rot = 0  ## indicates if image was rotated
    
    if orientation == 3:         ### otoceni kvuli spatnemu nacteni 
      img = rotateImage(img, 180)
      which_rot = 180
    elif orientation == 6:  
      img = rotateImage(img, 90)
      which_rot = 90
    elif orientation == 8:
      img = rotateImage(img, -90)
      which_rot = -90

    img = np.array(img)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 250, 250)
    #image = blur_image(edges, amount = 7)

    
    #### edges detection will be here
    
    left_chop = 0
    
    ####
    
    
    #lines = cv2.HoughLinesP(image, rho = 1, theta = math.pi, threshold = 20, minLineLength = img.shape[0]/1.5, maxLineGap = 0)
    #if lines is None:
    #    lines = cv2.HoughLinesP(image, rho = 1, theta = math.pi, threshold = 50, minLineLength = img.shape[0]/2.5, maxLineGap = 3)

    img_chopped = img.copy()
    
    
    
    ##### detection
    PERC = 30

    print('image ok')
    writelog('Image successfully prepared for recognition', env=env)

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()

    model = "{}temp_model.h5".format(modelpath)
    detector.setModelPath(model)

    writelog('1.', env=env)


    writelog('1.', env=env)
    detector.loadModel()

    print('model OK')
    writelog('Model successfully prepared.', env=env)

    custom_objects = detector.CustomObjects(person=False, car=False, bottle = True)
    main_obj, detections, extracted_obj = detector.detectCustomObjectsFromImage(input_image= img_chopped, 
                                                        input_type="array",
                                                        output_type = 'array', 
                                                        output_image_path= "{}im.png".format(outputpath), 
                                                        custom_objects=custom_objects,
                                                        extract_detected_objects=True,
                                                        minimum_percentage_probability=PERC)

    
      
    limit_det = 3   ### minimum bńumber of detections, if less then turn the image
    if len(detections)<=limit_det:
      writelog('Otoceni', env=env)
      img_chopped_rot = rotateImage(img_chopped, 90)
      main_obj_2, detections_2, extracted_obj_2 = detector.detectCustomObjectsFromImage(input_image= img_chopped_rot, 
                                                        input_type="array",
                                                        output_type = 'array', 
                                                        output_image_path= "{}im.png".format(outputpath), 
                                                        custom_objects=custom_objects,
                                                        extract_detected_objects=True,
                                                        minimum_percentage_probability=PERC)
       
      if len(detections_2) > limit_det:
        main_obj = main_obj_2
        detections = detections_2
        extracted_obj = extracted_obj_2
        img_chopped = img_chopped_rot
        which_rot = 90
      else:  
        img_chopped_rot = rotateImage(img_chopped, -90)
        main_obj_3, detections_3, extracted_obj_3 = detector.detectCustomObjectsFromImage(input_image= img_chopped_rot, 
                                                        input_type="array",
                                                        output_type = 'array', 
                                                        output_image_path= "{}im.png".format(outputpath), 
                                                        custom_objects=custom_objects,
                                                        extract_detected_objects=True,
                                                        minimum_percentage_probability=PERC)
        if len(detections_3) > limit_det:
            main_obj = main_obj_3
            detections = detections_3
            extracted_obj = extracted_obj_3
            img_chopped = img_chopped_rot
            which_rot = -90
            
       
    print(len(detections), 'detekci z prvni detekce')
    
    
    detections_all = detections.copy()
    extracted_all = extracted_obj.copy()
    img_chopped_black = img_chopped.copy()
    
    
        
    for i in range(len(detections)):
      img_chopped_black[detections[i]['box_points'][1]:detections[i]['box_points'][3],detections[i]['box_points'][0]:detections[i]['box_points'][2],:] = 0
    PERC = 5

    writelog('Second detection', env=env)

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet() 
    detector.setModelPath(model)

    detector.loadModel()

    custom_objects = detector.CustomObjects(person=False, car=False, bottle = True)
    main_obj, detections, extracted_obj = detector.detectCustomObjectsFromImage(input_image= img_chopped_black, 
                                                        input_type="array",
                                                        output_type = 'array', 
                                                        output_image_path="{}im.png".format(outputpath), 
                                                        custom_objects=custom_objects,
                                                        extract_detected_objects=True,
                                                        minimum_percentage_probability=PERC)
    for i in detections:        
      detections_all.append(i)
    for i in extracted_obj:   
      extracted_all.append(i)


    gc.collect()
    
    
##### filtering

  user = jsn['metadata']['Uploader-Name']
  store = jsn['metadata']['Store-Address']
  retailer = store.split(" ")[0]
  address = " ".join(store.split(" ")[1:])
  order = jsn['order']

  # order = 3
  # store = "Sárská 133/5 Praha Praha 13 15500 (S2753CZ)"

  ### load planograms and check if there is any big can in photo
  #   (for function filterDetections)

  plan_path = "{}parsed_planograms.pickle".format(modelpath)
  with open(plan_path, 'rb') as handle:
        planograms = pickle.load(handle)

  store_location = parse_store_location(store)
  date="2018-11"
  plan_dates = list(map(lambda x:x.split("_")[0],planograms["plan_name"]))
  plan_stores = list(map(lambda x:x.split("_")[2],planograms["plan_name"]))
  plan_names_df = pd.DataFrame({"plan_dates" : plan_dates, "plan_stores": plan_stores})
  target_ind = plan_names_df.index[(plan_names_df["plan_dates"] == date) & (plan_names_df["plan_stores"] == store_location)].tolist()[0]
  plan = planograms["plan_df"][target_ind]
  row, column = order_shelf_map(order)
  
  plan_all = pd.DataFrame()
  for i in range(0,3):  ### three shelves in one photo (ideally)
      row_pom = row + i
      plan_pom = plan.loc[(plan['row'] == row_pom) & (plan['column'] == column)]
      plan_all = plan_all.append(plan_pom)

  if max(plan_all['unit_height']) > 15:   ### big can is present (needed for heuristics)
    isBig = True
  else:
    isBig = False    

  ######### heuristics
    
  shelf_tuple = [(l['box_points'][0], l['box_points'][2]) for l in detections_all]
  shelf_tuple_y = [(l['box_points'][1], l['box_points'][3]) for l in detections_all]
  shelf_tuple_wo = shelf_tuple.copy()
  shelf_tuple.append((0,0))
  shelf_tuple.append((img_chopped.shape[1], img_chopped.shape[1]))
  #sorted_by_lower_bound = sorted(shelf_tuple, key=lambda tup: tup[0])
  # detections_orig = detections_all
  sizes_x = [l[1] - l[0] for l in shelf_tuple_wo]
  sizes_y = [l[1] - l[0] for l in shelf_tuple_y]
  sizes_x_arr = np.array(sizes_x)
  sizes_y_arr = np.array(sizes_y)
    
  print(len(detections_all))
  print(len(extracted_all))
    
  detections_wo, extracted_wo = filterDetections(sizes_x_arr, sizes_y_arr, detections_all, extracted_all, isBig)  ## filter weird boxes by coords

  print(len(detections_wo))
  print(len(extracted_wo))
  shelf_tuple = [(l['box_points'][0], l['box_points'][2]) for l in detections_wo]
  shelf_tuple_y = [(l['box_points'][1], l['box_points'][3]) for l in detections_wo]
  shelf_tuple_wo = shelf_tuple.copy()
  shelf_tuple.append((0,0))
  shelf_tuple.append((img_chopped.shape[1], img_chopped.shape[1]))
  #sorted_by_lower_bound = sorted(shelf_tuple, key=lambda tup: tup[0])

  sizes_x = [l[1] - l[0] for l in shelf_tuple_wo]
  sizes_y = [l[1] - l[0] for l in shelf_tuple_y]
  sizes_x_arr = np.array(sizes_x)
  sizes_y_arr = np.array(sizes_y)
 
  img_chopped_fin = img_chopped.copy()


# In[8]:

    ### version without horizontal edges detection
    
  GAP = 15
  detections = detect_shelves(copy.deepcopy(detections_wo)) # zaradi do regalu
  detections_sh = correct_shelves(copy.deepcopy(detections), img_chopped_fin) # oznaci ty co nejsou v zadnem regalu

    
    ### version with horizontal edges detection
    ## lines ... ouput from edge detection function
    
  hor_edg_det = False  ## delete this and the if after adding the function
    
  if hor_edg_det == True:
        lines.append(img.shape[0])
        lines_correct = []
        min_in_shelf = 2
        last_n_incorrect = 0
        for i in range(len(lines)): 
          if i == 0:  ## upper line
            upperLine = 0  ## top of image
            lowerLine = lines[0]
          else:  
            if last_n_incorrect > 0:   ### in this case, upperline will not move
              if last_n_incorrect >= i:   ### we would be oout of array (lines[-something])
                upperLine = 0
              else:
                upperLine = lines[i-last_n_incorrect-1]
            else: 
              upperLine = lines[i-1]
            
            lowerLine = lines[i]
          #print([upperLine, lowerLine])    
          cntr = 0 ### how many detected boxes is between
          for idx in range(len(detections_wo)): 
            if isInShelf(detections_wo[idx], upperLine, lowerLine) == True:
              cntr = cntr + 1
          #print(cntr)
          if cntr > min_in_shelf:
            lines_correct.append(lines[i])
            last_n_incorrect = 0
          else:
            last_n_incorrect = last_n_incorrect + 1
            
        ### we have correct lines of horizontal edged 
        ## match detected boxes to thosse shelves
        
        detections_sh = getShelves(copy.deepcopy(detections_wo), copy.deepcopy(lines_correct))
        

# In[9]:

  shelves = list()   ### filter out shelves with <= .. detected objects
  for i in range(len(detections_sh)):  
      shelves.append(detections_sh[i]['shelf'])  ## separate shelf numbers
  x=np.array(shelves)
  unique, counts = np.unique(x, return_counts=True)   ### frequency
    
  print(unique)
  print(counts)
    
  if unique[0]== -1:
      counts2= counts[1:len(counts)]   ### eliminate -1 shelf
      unique2= unique[1:len(unique)]
  else:
      counts2=counts
      unique2=unique

  shelves_ok = list()
  for i in range(len(counts2)):
      if counts2[i]>=4:   ### choose shelves
        shelves_ok.append(unique2[i])
  
  detections_ok = list()   ### filter detections in ok shelves (-1 removed)
  extracted_all_ok = list()
  for i in range(len(detections_sh)):
      if detections_sh[i]['shelf'] in shelves_ok:
        detections_ok.append(detections_sh[i])
        extracted_all_ok.append(extracted_wo[i])
 
   ### premapovani regalu 0:"jejich pocet"
  unique_po, counts_po = np.unique(np.array(shelves_ok), return_counts=True)
  seq=list()
  for i in range(len(unique_po)):
    seq.append(i)
  
  d = {unique_po[i]: seq[i] for i in range(len(seq))}
  
  detections_ok_reshelved = copy.deepcopy(detections_ok)
  for i in range(len(detections_ok_reshelved)):
    detections_ok_reshelved[i]['shelf'] = d.get(detections_ok[i]['shelf'])


  shlvs = list()
  for det in detections_ok_reshelved:
    shlvs.append(det['shelf'])


  img_chopped_completed = img_chopped_fin.copy()
  

# In[11]:


### extimate real measures

  can_rb_x = 7
  can_rb_y = 15

  true_sizes_x =  img_chopped_completed.shape[1]/np.median(sizes_x_arr)*can_rb_x
  true_sizes_y =  img_chopped_completed.shape[0]/np.median(sizes_y_arr)*can_rb_y

# memory
  #K.clear_session()
  #tf.reset_default_graph()


# In[12]:
######### recognition automl model ############
  
  writelog('AutoML calling...', env=env)

  #prediction_client = automl_v1beta1.PredictionServiceClient()
  #project_id = 'saleshouse-prototype'
  #model_id = 'ICN4760874677604180454' 
  #model_id = 'ICN7606729482481501366'
  #model_id = 'ICN8555118107281638010'
  #name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  ###################################################

  ######### load local recognition model ###############

  # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  #     http://www.apache.org/licenses/LICENSE-2.0
  
  #writelog('1.', env=env)
  model_file = "{}output_graph.pb".format(modelpath)
  label_file = "{}output_labels.txt".format(modelpath)
  #writelog('1.', env=env)
  graph_1 = load_graph(model_file)  
  
  #########################################################
  #writelog('1.', env=env)
  tmp = [(0,0) for i in range(len(detections_wo))]
  tmp_2 = [(0,0) for i in range(len(detections_wo))]
  tmp_3 = [(0,0) for i in range(len(detections_wo))]
  tmp_4 = [(0,0) for i in range(len(detections_wo))]
  tmp_5 = [(0,0) for i in range(len(detections_wo))]
  tmp_6 = [(0,0) for i in range(len(detections_wo))]
  tmp_7 = [(0,0) for i in range(len(detections_wo))]
  tmp_8 = [(0,0) for i in range(len(detections_wo))]
  tmp_9 = [(0,0) for i in range(len(detections_wo))]

  
  class_tuples = {'rb': tmp, 'rbsf':tmp_4, 'bigshock': tmp_2, 'rbblue':tmp_3, 'rbred': tmp_5, 'rbturned': tmp_6, 'unknown': tmp_7, 'bigshockrest': tmp_8, 'rbyellow': tmp_9 }
  class_names = class_tuples.keys()
  no_same = 0
  no_diff = 0
  no = 0

  

  for e in range(len(extracted_all_ok)):
   
    #if len(extractions_wo[e]) > 0: 
    if extracted_all_ok[e].size != 0: 
      try:
        #writelog('1.', env=env)
        pom = cv2.cvtColor(extracted_all_ok[e], cv2.COLOR_RGB2BGR) 
        #writelog('1.', env=env)
        cv2.imwrite('{}img1.jpg'.format(outputpath), pom)
        print('create file OK')
    
        #with open('{}img1.jpg'.format(outputpath), 'rb') as ff:
        #  content = ff.read()
        #payload = {'image': {'image_bytes': content }}
        #params = {}
        #try:
          #request = prediction_client.predict(name, payload, params)
          #detections_wo[e]['class'] = request.payload[0].display_name
        lbl, pst = local_recognizer('{}img1.jpg'.format(outputpath), graph_1)
        detections_ok[e]['class'] = lbl 
        if detections_ok[e]['class'] in class_names:

          #print(lbl)
          #print(request.payload[0].display_name)
          #print('---')

          class_tuples[detections_ok[e]['class']][e] = (detections_ok[e]['box_points'][0],detections_ok[e]['box_points'][2])
      except Exception as err:
        print("chyba v predikci", e)
        writelog(err, env=env, level='WARNING')
        writelog('Iteration n.{} failed.'.format(e), env=env, level='WARNING')
  
  print('automl OK')
  writelog('AutoML recognition completed.', env=env)

# In[13]:


### backlog
  
  #cv2.imwrite("./test_write.jpg", img_chopped_completed)
  #!gsutil cp "./test_write.jpg" "gs://saleshousephotos/test.jpg"
  #no_diff


# In[ ]:


### color the shelves based on recognition

  writelog('Data preparation and DB load running.', env=env)
  
  tmp = bucket.blob(photos).download_as_string()
  img3 = Img.open(io.BytesIO(tmp))
  img3 = np.array(img3)
  
  if which_rot != 0:
      img3 = rotateImage(img3, which_rot)
  
  #pred_colored = colour_shelves_prediction(detections_wo, img)
  detections_wo_shift = shiftDetections(copy.deepcopy(detections_wo), left_chop)
  
  pred_colored = colour_shelves_prediction(detections_wo_shift, img3)
  predicted_photo = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
  cv2.imwrite("{}recognised_photo.jpg".format(outputpath), predicted_photo)
  bucket_name = output_bucket_name
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob('recognised/' + jsn['name'])
  blob.upload_from_filename('{}recognised_photo.jpg'.format(outputpath), content_type = 'image/jpg')
  blob.make_public()

  det_colored = colour_shelves(detections_wo_shift, img)
  detected_photo = cv2.cvtColor(det_colored, cv2.COLOR_RGB2BGR)
  cv2.imwrite("{}detected_photo.jpg".format(outputpath), detected_photo)
  bucket_name = output_bucket_name
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob('detected/' + jsn['name'])
  blob.upload_from_filename('{}detected_photo.jpg'.format(outputpath), content_type = 'image/jpg')
  blob.make_public()

  pth = "https://storage.googleapis.com/" + bucket_name + "/recognised/" + jsn['name']
  pth_raw = "https://storage.googleapis.com/" + jsn["bucket"] + "/" + jsn['name']
  pth_det = "https://storage.googleapis.com/" + bucket_name + "/detected/" + jsn['name']
  #]Img.fromarray(pred_colored, 'RGB')


# In[ ]:

### calculate shelf ratio (percentage of shelf covered)
### and prepare to check planograms
  
  sz = img_chopped_completed.shape[1]
  shelves_no = max(shlvs)
  df = list()
  objects= list()  ## for planograms
      
  for k in range(shelves_no+1):
    #k = 3
    this_shelf = list()
    shelf_labels = list()    ### labels present in shelf
    shelf_positions = list()  ### positions of cans in this shelf (both for checking planograms)
    for clss in class_names:
      cntr = 0
      shelf_tuple = list()
      for idx, l in enumerate(detections_wo):    ## (or detections_ok, does not matter)
        if l['shelf'] == k:
          shelf_tuple.append(class_tuples[clss][idx])
          #if l['class'] == clss:
          cntr = cntr + 1

        
      sorted_by_lower_bound = sorted(shelf_tuple, key=lambda tup: tup[0])
      
      merged = []

      for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    #objects.append({'label':shelf_labels, 'location':shelf_positions})  ### first shelf will be on objects[0], second on 1 etc.
   
      if clss != 'rbturned':    ## rbturned, bigshockrest will be an exception (for planograms rbturned==redbull, bigshockrest == bigshock)
        if clss != 'bigshockrest':          
            shelf_labels.append(clss)
            shelf_positions.append(merged)
        else:
            bigshockrest = merged
      else:
        rbturned = merged
          
        
      this_shelf.append({'label': clss, 'cans': merged})
      
      zastoupeni = sum([i[1] - i[0] for i in merged])/sz
      plechovek = sum([i[1] - i[0] for i in merged])/np.median(sizes_x_arr)
      df.append({ 'shelf': k, 'class': clss, 'count_cans' : np.ceil(plechovek), 'count_boxes' : cntr,  'representation' : zastoupeni})
      #df_plan.append({ 'shelf': k, 'label': clss, 'cans':merged})
      
      #df.append({ 'shelf': k, 'class': clss, 'count_cans' : np.ceil(plechovek), 'representation' : zastoupeni})
      #print(clss, k, zastoupeni, np.ceil(plechovek))
    
    objects.append({'label':shelf_labels, 'location':shelf_positions})  ### first shelf will be on objects[0], second on 1 etc.
  
    objects[k]
    for i in range(len(objects[k]['label'])):  ### find redbull index, bigshockrest index  
      if objects[k]['label'][i] == 'rb': 
        ind_rb = i
        rb_all = objects[k]['location'][i]      
      if objects[k]['label'][i] == 'bigshock':
        ind_bigshock = i
        bigshock_all = objects[k]['location'][i]

    for i in range(len(rbturned)):   ## bind rb_all and rbturned
      rb_all.append(rbturned[i])
  
    for i in range(len(bigshockrest)):   ## bind bigshock and bigshockrest
      bigshock_all.append(bigshockrest[i])
  
  
    sorted_by_lower_bound = sorted(rb_all, key=lambda tup: tup[0])   ### and sort and choose distinct for all redbulls
  
    merged = []

    for higher in sorted_by_lower_bound:
      if not merged:
          merged.append(higher)
      else:
          lower = merged[-1]
          if higher[0] <= lower[1]:
              upper_bound = max(lower[1], higher[1])
              merged[-1] = (lower[0], upper_bound)  # replace by merged interval
          else:
              merged.append(higher)
    
    objects[k]['location'][ind_rb] = merged   ### write it ro rb --> rbturned and rb together 
  
    sorted_by_lower_bound = sorted(bigshock_all, key=lambda tup: tup[0])   ### and sort and choose distinct for all redbulls
  
    merged = []

    for higher in sorted_by_lower_bound:
      if not merged:
          merged.append(higher)
      else:
          lower = merged[-1]
          if higher[0] <= lower[1]:
              upper_bound = max(lower[1], higher[1])
              merged[-1] = (lower[0], upper_bound)  # replace by merged interval
          else:
              merged.append(higher)  
    objects[k]['location'][ind_bigshock] = merged   ### write it ro rb --> rbturned and rb together  
  
  #get store, user from photo metadata
  user = jsn['metadata']['Uploader-Name']
  store = jsn['metadata']['Store-Address']
  retailer = store.split(" ")[0]
  address = " ".join(store.split(" ")[1:])
    
  ## planogram check
  products = ["", "Red Bull", "Shock!", "Red Bull Red", "Red Bull Blue", "Red Bull Sugar Free", "Red Bull Bez Cukru"]
  labels = ["unknown", "rb", "bigshock", "rbred", "rbblue", "rbsf", "rbsf"]

  plan_path = "{}parsed_planograms.pickle".format(modelpath)
  with open(plan_path, 'rb') as handle:
    planograms = pickle.load(handle)
   
  
  row, column = order_shelf_map(order)  ## get columns from the foto

  
  plan_check_all = pd.DataFrame()
  for i  in range(len(objects)):
      store_loc = parse_store_location(store)
      plan_check_df = planogram_check(objects[i], planograms, row= i, column= column, store_location=store_loc, date="2018-11")
      plan_check_df["shelf"] = i
      plan_check_df= plan_check_df.rename(index=str, columns={"label": "class", "extra_share" : "planogram_extra_share", "missing_share" : "planogram_missing_share"})
      plan_check_df = plan_check_df.append(plan_check_df[plan_check_df['class'] == 'rb'])  ## copy line with rb..
      plan_check_df.iloc[-1, plan_check_df.columns.get_loc('class')] = 'rbturned'  ## .. and make it rbturned
      plan_check_df = plan_check_df.append(plan_check_df[plan_check_df['class'] == 'bigshock'])  ## and the same with bigshockrest
      plan_check_df.iloc[-1, plan_check_df.columns.get_loc('class')] = 'bigshockrest'  
    
      plan_check_all = plan_check_all.append(plan_check_df)
          
    
  
 
  output_DF = pd.DataFrame(df)    

  output_DF['url_photo'] = pth_raw
  output_DF['url_photo_detected'] = pth_det
  output_DF['url_photo_recognised'] = pth
  output_DF['id_photo'] = int(jsn['generation'])
  output_DF['unique_id'] = output_DF["id_photo"].map(str) + output_DF["class"] + output_DF['shelf'].map(str)
  #output_DF['unique_id'] = output_DF["id_photo"].map(str)  + output_DF['shelf'].map(str)
  output_DF['shelf_sizes_est_x'] = int(true_sizes_x)
  output_DF['shelf_sizes_est_y'] = 15
  output_DF['representation'] = round(output_DF['representation'], 2)
  output_DF['retailer'] = retailer
  output_DF['store'] = address
  output_DF['user'] = user
  output_DF['shelf_column'] = 0
  #output_DF['planogram_eq_ratio'] = 'init'
  #output_DF['planogram_eq_ratio_extra'] = 'init'
  #output_DF['planogram_share'] = 0
  output_DF['after_completation'] = 0
  output_DF['photo_captured'] = photo_load
  output_DF['photo_load'] = photo_load
  output_DF['recognition_start'] = recognition_start
  output_DF['load_type'] = load_type


  output_DF = pd.merge(output_DF, plan_check_all, how='left', on=['class', 'shelf'])
  output_DF['planogram_share'] = output_DF['planogram_share'].fillna(0)  ### replace NA by 0 (were not present in recognised photo)
  output_DF['planogram_missing_share'] = output_DF['planogram_missing_share'].fillna(0)
  output_DF['planogram_extra_share'] = output_DF['planogram_extra_share'].fillna(0)


# In[ ]:


#Img.fromarray(img_chopped_fin, 'RGB')


# In[27]:


### write to bigquery
  
  print('writing to BQ')

  client = bq.Client()
  dataset_id = bq_dataset
  table_id = 'recognition_output'

  def uploaddata(dataset_id, table_id, rows):
      table_ref = client.dataset(dataset_id).table(table_id)
      table = client.get_table(table_ref)
      errors = client.insert_rows(table, rows)

      assert errors == []

  output_DF['count_boxes'] = output_DF['count_boxes'].map(int)
  output_DF['db_insert'] = datetime.datetime.now(timezone('Europe/Berlin')).strftime    ("%Y-%m-%dT%H:%M:%S.%fZ")
 

  #url_photo_detected
  sorted_DF = output_DF[['retailer', 'store', 'user', 'unique_id', 'id_photo', 'url_photo', 'url_photo_recognised',
                         'url_photo_detected', 'shelf', 'class', 'representation', 'count_boxes', 'count_cans',
                         'shelf_sizes_est_x', 'shelf_sizes_est_y', 'shelf_column', 'planogram_missing_share',
                         'planogram_extra_share', 'planogram_share', 'after_completation', 'photo_captured', 
                         'photo_load', 'recognition_start', 'db_insert', 'load_type']]

  
  tuples = [tuple(x) for x in sorted_DF.values]   
  uploaddata(dataset_id, table_id, tuples)


# In[ ]:


#from datetime import datetime
  #datetime_object = datetime.strptime('2018-11-21T14:03:42.757Z', '%Y-%m-%dT%H:%M:%S.%fZ')
  
  del(main_obj)  
  del(detections)
  del(detections_all)
  del(extracted_obj)
  del(extracted_all)
  
  gc.collect()
  print('OK')


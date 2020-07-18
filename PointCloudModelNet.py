import torch
import zipfile
import os
import tqdm
import random
import numpy as np

def gen_bar_update():
  prog_bar = tqdm.autonotebook.tqdm(total=None)
  def bar_update(count,blocksize,totalsize):
    if prog_bar.total is None and totalsize:
      prog_bar.total = totalsize
    progressbts = count * blocksize; 
    prog_bar.update(progressbts - prog_bar.n)
  return bar_update


class PointCloud(torch.utils.data.Dataset):
  """Attempts to provide a container for point cloud data. Still in progress"""
  
  
  
  
  #useful helpers lifted from mnist torchvision
  @property
  def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

  @property
  def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

  @property
  def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

  @property
  def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

 

  def __init__(self,
               root     : str,
               train    : bool = True,
               download : bool = True, 
               large    : bool = False, 
               verbose  : bool = False,
               per_dict : bool = False):

    super(PointCloud,self).__init__()

    location10 = "http://3dshapenets.cs.princeton.edu/ModelNet10.zip"
    location40 = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

    if large is True: 
      location = location40
    else:
      location = location10
    
    self.root = root
    self.model_size = ("40" if large else "10")

    self.available = []
    self.dataN = 0

    self.classes = ['laptop','person','bowl','night_stand','dresser','stairs','lamp',
      'flower_pot','radio','door','desk','cone','tent','xbox','sink','bookshelf',
      'wardrobe','bottle','monitor','piano','toilet','glass_box','bench','mantel',
      'vase','tv_stand','range_hood','bathtub','chair','curtain','plant','sofa',
      'table','stool','bed','guitar','cup','airplane','keyboard','car']

    if download:
      self.download_and_extract(root = self.root, filename = None, url = location)
      self.build_out(per_dict = per_dict)

    else:
      for clazz in self.classes:
        try:
          self.available = os.listdir(self.processed_folder)
          self.dataN = len(self.available)

        except FileNotFoundError: 
          print(f"class {clazz} unsuccessfully loaded, continuing")
  
  #Useful helpers lifted from mnist torchvision
  @property
  def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

  @property
  def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

  @property
  def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

  
  def extract(self, archive, to_path):
    with zipfile.ZipFile(archive,"r") as z:
      z.extractall(to_path)
  
  def download_and_extract(self,root,filename,url):
    if filename is None:
      filename = os.path.basename(url)
    to_path = os.path.dirname(root)

    self.download(root,filename = filename, url = url)
    archive = os.path.join(root, filename)
    self.extract(archive, to_path)
  
  def download(self,root,filename,url): 
    import urllib
    
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True);
    
    fpath = os.path.join(root, filename)

    try:
      urllib.request.urlretrieve(url, fpath, reporthook = gen_bar_update())
    except (urllib.error.URLError, IOError) as e:
      raise e


  def load_cloud(self, path):
    """parse point cloud data from .off extension, still a bit slow"""

    if self.model_size is "10":
      with open(path,"rb") as f:
        x = f.read()
        x = x.decode("utf-8")
        x = x.split("\n")
    else:
      with open(path,"r") as f:
        x=f.readlines()

    try:
      counts = list(map(int, x[1].strip().split(" ")))

      points = np.array([list(map(float, x[i].strip().split(" "))) 
                    for i in range(2,counts[0]+2)])
      indices = [list(map(int, x[i].strip().split(" ")[1:])) 
                    for i in range(2+counts[0], counts[1]+2+counts[0])]

      return points, indices #map_vertices(points, indices)
    
    except Exception as e: 
      raise e

    
  def build_out(self, train : bool = True, per_dict : bool = False):
    """Construct all data in terms of python native containers for fast load at train time

       @per_dict: save a dictionary of all class instances, keys are instance names, default false.
       (default is save all items in same directory and load them all at runtime)

       @train: train data or test data. only train is implemented, and is only implemented for model net 40
    """


    dir = os.path.join(self.root, "ModelNet" + self.model_size)
    categories = self.classes
    cat_dict = {}

    if train:
      for category in categories:

        cat_top = os.path.join(dir, category)
        training = os.path.join(cat_top, "train")
        dir_cat = os.listdir(training)

        if not os.path.isdir(self.processed_folder):
          if not os.path.isdir( os.path.join( self.root, self.__class__.__name__)):
            os.mkdir( os.path.join( self.root, self.__class__.__name__))
          os.mkdir(self.processed_folder)

        cat_path = os.path.join(self.processed_folder, category)

        for instance in dir_cat:

          if category not in cat_dict.keys():
            cat_dict[category] = {}   

          try:
            if per_dict:
              cat_dict[category][instance.split("_")[-1]] \
                 = self.load_cloud(os.path.join(training, instance))
            else:
              data = self.load_cloud(os.path.join(training, instance))
              label = self.classes.index(category)


              torch.save((data,label),
                          os.path.join(self.processed_folder, instance) + ".pt")
            self.dataN += 1

          except ValueError as ve:
             print(f"instance {instance} improperly formatted for load_cloud, with error {ve}, continuing...")
          

        if per_dict:
          torch.save(cat_dict[category], cat_path + ".pt")



        #Still wondering if I will ever have a reason to maintain all 40 classes in memory
        cat_dict[category] = None # free references, will take up so much ram otherwise
        print(f"saved {category} to {cat_path}, option per dict is {per_dict}")
  


  def __getitem__(self, index):
    d, t  = torch.load(
              os.path.join(self.processed_folder, self.available[index]))
    
    t = torch.tensor(t)
    d = torch.tensor(d[0],dtype=torch.float32) #discard surface description data

    dout = d.reshape(-1,1,3).cuda() #reshape to have a tensor of only 3d coords

    dout = (dout - dout.min(-3)[0])/(dout.max(-3)[0] - dout.min(-3)[0]) #normalize (will try and center it as well, as paper describes this)
    
    d_count = dout.shape[0]
    permute = [i for i in range(d_count)]; random.shuffle(permute)# permute ordering of vertices

    if d_count >= 5000:
      permuted = torch.ones((5000,1,3),dtype=torch.float32).cuda() * dout[torch.tensor(permute[:5000]).cuda()] #cap at 5000 points, also described
    else: 
      permuted = torch.ones_like(dout).cuda()*dout[torch.tensor(permute).cuda()] #otherwise, not need to cap

    return permuted, t


  def __len__(self):
    return len(self.available)

  """
  def check_exists(self):
    return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))"""
  
  

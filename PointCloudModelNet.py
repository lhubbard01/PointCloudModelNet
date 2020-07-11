import zipfile
import os
import tqdm
import numpy as np
import torchvision
import torch





def gen_bar_update():
  prog_bar = tqdm.autonotebook.tqdm(total=None)
  def bar_update(count,blocksize,totalsize):
    if prog_bar.total is None and totalsize:
      prog_bar.total = totalsize
    progressbts = count * blocksize; 
    prog_bar.update(progressbts - prog_bar.n)
  return bar_update


class PointCloud(torch.utils.data.Dataset):
  training_file = 'training.pt'
  test_file = 'test.pt'
   

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

  def __init__(self,root:str,train:bool = True,download : bool = True, large : bool = False):
    super(PointCloud,self).__init__()
    location10 = "http://3dshapenets.cs.princeton.edu/ModelNet10.zip"
    location40 = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    if large is True: 
      location = location40
    else:
      location = location10
    self.root = root
    self.model_size = ("40" if large else "10")
    if download and not self.check_exists():
      self.download_and_extract(root = self.root, filename = None, url = location)
    else:
      pass
    
    self.build_out()
  
  def check_exists(self):
    return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))
  
  
  def load_cloud(self, path):

    if self.model_size is "10":
      with open(path,"rb") as f:
        x = f.read()
        x = x.decode("utf-8")
        x = x.split("\n")
    else:
      with open(path,"r") as f:
        x=f.readlines()
    counts = list(map(int, x[1].strip().split(" ")))
    points = np.array([list(map(float, x[i].strip().split(" "))) for i in range(2,counts[0]+2)])
    indices = [list(map(int, x[i].strip().split(" ")[1:])) for i in range(2+counts[0], counts[1]+2+counts[0])]
    return points, indices #map_vertices(points, indices)
  def build_out(self,train : bool = True):
    dir = os.path.join(self.root, "ModelNet" + self.model_size)
    categories = os.listdir(dir)
    cat_dict = {}
    if train:
      cnt = 0
      for category in categories:
        if cnt == 10: break
        cat_top = os.path.join(dir, category)
        training = os.path.join(cat_top, "train")
        dir_cat = os.listdir(training)
        for instance in dir_cat:
           if category not in cat_dict.keys():
             cat_dict[category] = {}
           else:
             cat_dict[category][instance.split("_")[-1]] = self.load_cloud(os.path.join(training, instance))
        print(f"{category} complete!"); cnt+=1



    self.data = cat_dict
  

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

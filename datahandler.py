import torch
import logging
import numpy as np
logger = logging.getLogger(__name__)


class BatchGenerator:
   def __init__(self,filelist,config,type):

      self.filelist     = np.array(filelist)
      self.evt_per_file = evt_per_file
      self.batch_size   = batch_size
      self.img_shape    = img_shape   # (channel,height,width)
      self.grid_shape   = grid_shape  # (h,w)
      self.num_classes  = num_classes
      self.rank         = rank
      self.nranks       = nranks
      self.use_random   = use_random

      self.total_images = self.evt_per_file * len(self.filelist)
      self.total_batches = self.total_images // self.batch_size // self.nranks
      self.batches_per_file = self.evt_per_file // self.batch_size

      self.files_per_rank = int(len(self.filelist) / self.nranks)

   def set_random_batch_retrieval(self,flag=True):
      self.use_random = flag

   def __len__(self):
      return self.total_batches

   def batch_gen(self):
      if self.use_random:
         np.random.shuffle(self.filelist)

      start_file_index = self.rank * self.files_per_rank
      end_file_index = (self.rank + 1) * self.files_per_rank

      logger.warning('rank %s processing files %s through %s',self.rank,start_file_index,end_file_index)
      logger.warning('first file after shuffle: %s',self.filelist[0])
      image_counter = 0
      raw_coords = None
      raw_features = None
      truth_batch = None

      for filename in self.filelist[start_file_index:end_file_index]:

         file = FileGenerator(filename,self.img_shape,self.grid_shape)

         if self.use_random:
            file.set_random_image_retrieval()

         for input_data in file.image_gen():
            single_raw,single_truth = input_data

            raw_coords,raw_features = self.merge_raw(single_raw,raw_coords,raw_features,image_counter)
            truth_batch = self.merge_truth(single_truth,truth_batch)
            # logger.info('truth_batch: %s',truth_batch.shape)

            image_counter += 1

            if image_counter == self.batch_size:
               yield {'images': [raw_coords,raw_features],'truth': truth_batch}

               raw_coords     = None
               raw_features   = None
               truth_batch    = None
               image_counter  = 0

   def merge_raw(self,raw,raw_coords,raw_features,image_counter):
      #raw = torch.from_numpy(raw)
      # create coords that includes image_count
      new_raw_coords = torch.zeros([len(raw[1]),3]).long()
      new_raw_coords[...,0:2] = torch.from_numpy(raw[1])
      new_raw_coords[...,2]   = torch.full((len(raw[1]),),image_counter)

      # convert features to torch tensor
      new_raw_features = torch.from_numpy(raw[0]).float()

      # merge new features to list

      if raw_coords is None:
         raw_coords = new_raw_coords.long()
         raw_features = new_raw_features.float()
      else:
         raw_coords = torch.cat([raw_coords,new_raw_coords])
         raw_features = torch.cat([raw_features,new_raw_features])

      return raw_coords,raw_features

   def merge_truth(self,truth,truth_batch):
      # convert truth to torch
      new_truth = torch.from_numpy(truth[np.newaxis,...]).double()

      if truth_batch is None:
         truth_batch = new_truth
      else:
         truth_batch = torch.cat([truth_batch,new_truth])

      return truth_batch


class FileGenerator:
   def __init__(self,filename,img_shape,grid_shape):
      self.filename     = filename
      self.img_width    = img_shape[2]
      self.img_height   = img_shape[1]
      self.grid_w       = grid_shape[1]
      self.grid_h       = grid_shape[0]

      self.use_random = False

   def open_file(self):
      try:
         logger.info('opening file: %s',self.filename)
         nf = np.load(self.filename,allow_pickle=True)
         self.raw = nf['raw']
         truth = nf['truth']

         # a = time.time()
         self.truth = convert_truth(truth,self.img_width,self.img_height,self.grid_w,self.grid_h)
         # logger.info('convert_truth time: %s',time.time() - a)
      except:
         logger.exception('exception received when opening file %s',self.filename)
         raise

   def __getitem__(self,idx):
      if not hasattr(self,'raw'):
         self.open_file()
      assert(idx < len(self.raw))
      return (self.raw[idx],self.truth[idx])

   def set_random_image_retrieval(self,flag=True):
      self.use_random = flag

   def image_gen(self):
      if not hasattr(self,'raw'):
         self.open_file()

      index_list = np.arange(len(self.raw))

      if self.use_random: np.random.shuffle(index_list)
      for idx in index_list:
         yield (self.raw[idx],self.truth[idx])


def convert_truth(intruth,img_width,img_height,grid_w,grid_h,new_channels=2):
   pix_per_grid_w = img_width / grid_w
   pix_per_grid_h = img_height / grid_h

   intruth_size = len(intruth)

   new_truth = np.zeros((intruth_size,new_channels,grid_h,grid_w),dtype=np.int32)

   for img_num in range(intruth_size):

      img_truth = intruth[img_num]

      for obj_num in range(len(img_truth)):
         obj_truth = img_truth[obj_num]

         obj_exists = obj_truth[0]
         
         # logger.info('%s = %s %s',img_num,obj_exists,obj_truth[5:])

         if obj_exists == 1:

            obj_center_x = obj_truth[1] / pix_per_grid_w
            obj_center_y = obj_truth[2] / pix_per_grid_h
            # obj_width    = obj_truth[3] / pix_per_grid_w
            # obj_height   = obj_truth[4] / pix_per_grid_h

            grid_x = int(np.floor(obj_center_x))
            grid_y = int(np.floor(obj_center_y))
            # logger.info('%s = %s %s    %s',img_num,grid_x,grid_y,obj_truth[1:5])

            if grid_x >= grid_w:
               raise Exception('grid_x %s is not less than grid_w %s' % (grid_x,grid_w))
            if grid_y >= grid_h:
               raise Exception('grid_y %s is not less than grid_h %s' % (grid_y,grid_h))

            if new_truth[img_num,0,grid_y,grid_x] == 1:
               logger.warning('two ojects in the same grid found: image num = %s, image truth: \n%s',img_num,np.int32(img_truth))
               continue

            new_truth[img_num,0,grid_y,grid_x] = obj_exists
            new_truth[img_num,1,grid_y,grid_x] = np.argmax([np.sum(obj_truth[5:10]),
                                                            np.sum(obj_truth[10:12])
                                                           ]
                                                          )

      if img_truth[:,0].sum() != new_truth[img_num,0,:,:].sum():
         logger.warning('images have different content: %s != %s',img_truth[:,0].sum(),new_truth[img_num,0,:,:].sum())

   logger.info('new_truth: %s  %s   %s',new_truth.shape,new_truth[:,0,:,:].sum(axis=1).sum(axis=1).mean(),new_truth.sum())
   return new_truth
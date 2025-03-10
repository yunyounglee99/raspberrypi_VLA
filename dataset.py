import torch
import os
import json
import torchvision.transforms as transforms
import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset

class OpenVLADataset(Dataset):
  def __init__(self, data_dir, tokenizer, transform = None):
    self.data_dir = data_dir
    self.tokenizer = tokenizer
    self.transform = transform if transform else transforms.ToTensor()

    self.metadata_file = os.path.join(data_dir, "metadata.json")
    with open(self.metadata_file, "r") as f:
      self.metadata = json.load(f)

    self.keys = list(self.metadata.keys())

  def __len__(self):
    return len(self.keys)
  
  """load data part는 필요할까 싶긴함(아닐수도)"""
  def load_image(self, image_path):
    image = Image.open(image_path).convert("RGB")
    return self.transform(image)
  
  def load_text(self, text):
    encoded_text = self.tokenizer(text, padding = "max_length", truncation=True, return_tensors = "pt")
    return encoded_text["input_ids"].squeeze(0), encoded_text["attention_mask"].squeeze(0)
  
  def load_lidar(self, lidar_path):
    pcd = o3d.io.read_point_cloud(lidar_path)
    points = np.asarray(pcd.points, dtype = np.float32)
    return torch.tensor(points)
  
  def __getitem__(self, idx):
    key = self.keys[idx]
    sample = self.metadata[key]

    image = self.load_image(os.path.join(self.data_dir, sample["image_path"]))
    text_input, text_mask = self.load_text(sample["text"])
    lidar = self.load_lidar(os.path.join(self.data_dir, sample["lidar_path"]))

    return {
      "image" : image,
      "text_input" : text_input,
      "text_mask" : text_mask,
      "lidar" : lidar
    }
  
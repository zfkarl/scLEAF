import torch
import torch.utils.data as data
import numpy as np
import scipy.sparse
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.demos.boring_classes import RandomDataset
import scanpy as sc
import anndata
import scvi
from sklearn import preprocessing
def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
    return np.loadtxt(label_file)

def npz_reader(file_name):
    print('load npz matrix:', file_name)
    data = scipy.sparse.load_npz(file_name)
    adata = anndata.AnnData(X=data)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    return adata.X
        
class Cell_Text_Dataset(data.Dataset):
    def __init__(self, data_path = None, text_path = None, label_path = None, gene_path = None):    

        self.data_reader= npz_reader(data_path) 
        self.text_emb = np.load(text_path)  
        self.labels = load_labels(label_path)
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        self.text_emb_size = self.text_emb.shape[1]
        self.gene_emb = np.load(gene_path)
        self.gene_emb_size = self.gene_emb.shape[1]
        
        self.gene_emb_per_cell = np.array(self.data_reader@self.gene_emb).astype('float32')
 
        #print('gene emb shape: ', self.gene_emb_per_cell.shape, self.gene_emb_per_cell.dtype)
        
        assert self.text_emb.shape[0] == self.data_reader.shape[0]
        
    def __getitem__(self, index):

        sample = np.array(self.data_reader[index].todense())
        cell= sample.reshape((1, self.input_size)).astype('float32')
        text = self.text_emb[index]
        label = self.labels[index].astype(int)

        gene_text = self.gene_emb_per_cell[index]
        
        return cell, text, gene_text, label

    def __len__(self):
        return self.sample_num
                

class CellTextDataModule(pl.LightningDataModule):
    def __init__(self,batch_size = 512, data_path = None, text_path = None, label_path = None, gene_path = None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.text_path = text_path
        self.label_path = label_path
        self.gene_path = gene_path
        
        self.dataset = Cell_Text_Dataset(self.data_path,self.text_path,self.label_path,self.gene_path)

        n_train, n_val = int(0.8*len(self.dataset)), int(0.1*len(self.dataset))
        n_test = len(self.dataset) - n_train - n_val

        self.train, self.val, self.test = data.random_split(
            self.dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size = self.batch_size, drop_last=False,num_workers=4)
    
    
    
    
class Cell_Text_Dataset_PBMC(data.Dataset):
    def __init__(self, text_path = None,  gene_path = None):    

        self.adata = scvi.data.pbmc_dataset() 
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.text_emb = np.load(text_path)  
        self.labels = self.adata.obs['labels']
        self.input_size = self.adata.X.shape[1]
        self.sample_num = self.adata.X.shape[0]
        self.text_emb_size = self.text_emb.shape[1]
        self.gene_emb = np.load(gene_path)
        self.gene_emb_size = self.gene_emb.shape[1]
        
        self.gene_emb_per_cell = np.array(self.adata.X@self.gene_emb).astype('float32')
 
        assert self.text_emb.shape[0] == self.adata.X.shape[0]
        
    def __getitem__(self, index):

        sample = np.array(self.adata.X[index].todense())
        cell= sample.reshape((1, self.input_size)).astype('float32')
        text = self.text_emb[index]
        label = self.labels[index].astype(int)

        gene_text = self.gene_emb_per_cell[index]
        
        return cell, text, gene_text, label

    def __len__(self):
        return self.sample_num

        
class CellTextDataModule_PBMC(pl.LightningDataModule):
    def __init__(self,batch_size = 512, text_path = None,  gene_path = None):
        super().__init__()
        self.batch_size = batch_size
        self.text_path = text_path
        self.gene_path = gene_path
        
        self.dataset = Cell_Text_Dataset_PBMC(self.text_path,self.gene_path)

        n_train, n_val = int(0.9*len(self.dataset)), int(0.1*len(self.dataset))
       
        self.train, self.val = data.random_split(
            self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def test_dataloader(self):
        return data.DataLoader(self.datasest, batch_size = self.batch_size, drop_last=False,num_workers=4)
    
    
class Cell_Text_Dataset_BMMC(data.Dataset):
    def __init__(self, text_path = None,  gene_path = None):    

        self.adata = sc.read('/data2/zeyu/zeyu1/dataset/sc-data/data/BMMC_multiomics.h5ad')
        self.text_emb = np.load(text_path)  
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.adata.obs['celltype'].values)
        self.input_size = self.adata.X.shape[1]
        self.sample_num = self.adata.X.shape[0]
        self.text_emb_size = self.text_emb.shape[1]
        self.gene_emb = np.load(gene_path)
        self.gene_emb_size = self.gene_emb.shape[1]
        
        self.gene_emb_per_cell = np.array(self.adata.X@self.gene_emb).astype('float32')
 
        assert self.text_emb.shape[0] == self.adata.X.shape[0]
        
    def __getitem__(self, index):

        sample = np.array(self.adata.X[index].todense())
        cell= sample.reshape((1, self.input_size)).astype('float32')
        text = self.text_emb[index]
        label = self.labels[index].astype(int)

        gene_text = self.gene_emb_per_cell[index]
        
        return cell, text, gene_text, label

    def __len__(self):
        return self.sample_num
    
    
class CellTextDataModule_BMMC(pl.LightningDataModule):
    def __init__(self,batch_size = 512, text_path = None,  gene_path = None):
        super().__init__()
        self.batch_size = batch_size
        self.text_path = text_path
        self.gene_path = gene_path
        
        self.dataset = Cell_Text_Dataset_BMMC(self.text_path,self.gene_path)

        n_train = int(0.9*len(self.dataset))
        n_val = len(self.dataset) - n_train
       
        self.train, self.val = data.random_split(
            self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size = self.batch_size, drop_last=False,num_workers=4)

    def test_dataloader(self):
        return data.DataLoader(self.datasest, batch_size = self.batch_size, drop_last=False,num_workers=4)
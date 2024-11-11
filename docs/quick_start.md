# Quick Start

### Set up a new virtual environment
```bash
conda create -n instanceocc python=3.8 -y
conda activate instanceocc 
```

### Install packpages using pip3
```bash
instanceocc_path="path/to/instanceocc"
cd ${instanceocc_path}
pip3 install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install pyg_lib torch_scatter torch_sparse torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install einops
pip3 install -r requirement.txt
apt-get update && apt-get install -y ninja-build

### Compile the deformable_aggregation and others CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../deform_attn_3d 
python setup.py build_ext --inplace
cd ../projects/mmdet3d_plugin/models/bev_pool_v2
python setup.py build_ext --inplace
cd ../DFA3D
bash setup.sh
cd ../../../..
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.
```bash
cd ${sparse4d_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required .pkl files.
```bash
pkl_path="data/nuscenes_anno_pkls"
mkdir -p ${pkl_path}
python3 tools/create_data_instanceocc.py --version v1.0-mini --info_prefix ${pkl_path}/nuscenes-mini
python3 tools/create_data_instanceocc.py --version v1.0-trainval,v1.0-test --info_prefix ${pkl_path}/nuscenes
```

### Generate anchors by K-means
```bash
python3 tools/anchor_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing
```bash
# train
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704

# test
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704  path/to/checkpoint
```

For inference-related guidelines, please refer to the [tutorial/tutorial.ipynb](tutorial/tutorial.ipynb).

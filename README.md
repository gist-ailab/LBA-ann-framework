# LBA ann Framework

### Enviroment setting
```
# conda env create (sam2 needed python version uppper 3.10)
conda create -n lba python=3.10
conda activate lba

# pytorch install
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118


# mmdetection install
pip install -U openmim
mim install mmengine
install cuda 11.8 from https://developer.nvidia.com/cuda-11-8-0-download-archive
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
cd lib
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection 
pip install -v -e .


# sam2 install

cd lib
git clone https://github.com/facebookresearch/sam2.git
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \

# requirments
pip install opencv-python
pip install Pyside6
pip install imgviz
pip install scikit-image
pip install fancy
pip install -U scikit-learn

```
### Code run
```
python main.py
```


### ToDo List
- prompt labeling function using sam2
- auto labeling function using training model
- coco json read function
- annotation save function
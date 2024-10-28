# LBA-ann-framework

### Enviroment setting
```
# pytorch install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# sam2 install
cd segment_anything2
pip install -e .

# requirments
pip install opencv-python
pip install Pyside6
pip install imgviz
pip install scikit-image

```

### ToDo List
- prompt labeling function using sam2
- auto labeling function using training model
- coco json read function
- annotation save function
This is the official implementation of the paper 'Tackling the Background Bias in Sparse Object Detection via Cropped Windows'
Here you can find a Pytorch data set which reduces the background bias for sparse recordings and allows the usage of higher resolutions during training.
The technique is only applicable the object detection.

### Description
It is implemented as a data set wrapper (see <em>CroWTiledDataset</em>), which accepts a data set defining. 
This data set should implement the <em>BaseDataset</em> interface. 

### Example
1. Install the requirements.txt
2. Execute "python3 example.py"
3. output/ contains the created tiles. (All classes are visualized with green bounding boxes)

In the default configuration, 512x512 tiles are created. 
Further the full frame is added with a downscaling factor of 0.5.
Empty tiles were discarded.

### Folder structure
 - 'crow_dataset.py' contains the data set wrapper. This is the only part necessary for CroW 
 - 'example.py' shows the usage of the tiling data set on an example MS COCO format data set
 - 'example' contains an example dataset in MS COCO format
 - 'output' will contain the created tiles with bounding boxes. (All classes are visualized with green bounding boxes)


### Citation
```
@InProceedings{Varga_2021_ICCV,
    author    = {Varga, Leon Amadeus and Zell, Andreas},
    title     = {Tackling the Background Bias in Sparse Object Detection via Cropped Windows},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2768-2777}
}
```

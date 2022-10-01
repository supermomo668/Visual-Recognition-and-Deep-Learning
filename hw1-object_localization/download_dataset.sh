# First, cd to a location where you want to store ~0.5GB of data.
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
cd data
tar xf VOCtrainval_06-Nov-2007.tar
# Also download the test data
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
export DATA_DIR=$(pwd)
cd VOCdevkit/VOC2007/
# Download the selective search data from https://drive.google.com/drive/folders/1jRQOlAYKNFgS79Q5q9kfikyGE91LWv1I to this location
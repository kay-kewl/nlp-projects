#!/bin/bash
mkdir -p data
cd data

# Data for seq2seq translation
echo "Downloading translation data"
wget https://www.dropbox.com/s/yy2zqh34dyhv07i/data.txt?dl=1 -O data.txt

# Data for BSNLP
echo "Downloading BSNLP data"
wget https://bsnlp.cs.helsinki.fi/bsnlp-2019/SAMPLEDATA_BSNLP_2019_shared_task.zip
unzip SAMPLEDATA_BSNLP_2019_shared_task.zip

# Data for AG News
echo "Downloading AG News Data..."
wget https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv -O ag_news_train.csv
wget https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv -O ag_news_test.csv

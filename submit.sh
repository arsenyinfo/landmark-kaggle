#!/usr/bin/env bash

rm result/submit.zip
zip result/submit.zip result/pred.csv
kaggle competitions submit -c landmark-recognition-challenge -f result/submit.zip -m "just another submit"

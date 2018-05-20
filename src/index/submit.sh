#!/usr/bin/env bash

rm result/retrieval.zip
zip result/retrieval.zip result/retrieval.csv
kaggle competitions submit -c landmark-retrieval-challenge -f result/retrieval.zip -m "just another submit"

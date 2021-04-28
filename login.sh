#!/bin/bash

gcloud auth login
gcloud auth application-default login
gcloud config set project sunlit-analyst-309609

mkdir -p data
gcsfuse --implicit-dirs --limit-bytes-per-sec -1 --limit-ops-per-sec -1 arielml_data data
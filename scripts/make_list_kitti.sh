#!/bin/sh
find dataset/ -name "*.png" > dataset/train_list.txt
find dataset/2011_09_26/2011_09_26_drive_0001_extract/ -name "*.png" > dataset/test_list.txt

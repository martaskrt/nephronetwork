#!/bin/bash

in_folder="C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/Full_dataset/"
script_folder="C:/Users/lauren erdman/Desktop/kidney_img/HN/nephronetwork/DA/OtherSites/Stanford/"
out_folder="C:/Users/lauren erdman/Desktop/kidney_img/HN/Stanford/std_jpgs0/"

ls "${in_folder}"*.tgz > "${in_folder}"/file_list.txt

cd "${in_folder}"

while read eachfile
do
  eachfile_root="my_dcm"
  
  mkdir -p $eachfile_root

  new_test_file="${eachfile##*/}"
  echo "File being unzipped:"
  echo $new_test_file

  tar -xf $new_test_file --strip-components=1 -C $eachfile_root

  echo "Extracting dicom images"
  python "${script_folder}"/stan_prep0_v0.py -img_dir "${in_folder}"/$eachfile_root -out_dir "${out_folder}" #-fileroot $eachfile_root

  ### strip filename down to root (or just pass to python script to do this)
  ## pass through python script and have it write black and white images + csvs
  #                                                 to out_folder (pass out_folder as argument)

  rm -r "${in_folder}"/$eachfile_root

  

done < "${in_folder}"/file_list.txt
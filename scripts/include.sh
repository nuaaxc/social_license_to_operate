#!/usr/bin/env bash
hostname=$(hostname -f)
src_name="social_license_to_operate"
project_name="social_license_to_operate"


if [[ $hostname == "Bindi-EP" ]] || [[ $hostname == "chang" ]]
then
  src_path="$HOME/PycharmProjects/$src_name"
  data_dir="$HOME/Dropbox/project/$project_name/data/"
  word_vec_dir="$HOME/Dropbox/resources/pretrained/word_vec/"
elif [[ $hostname == *"alienware-ep.nexus.csiro.au"* ]]
then
  src_path="$HOME/project/$project_name/$src_name"
  data_dir="$HOME/project/$project_name/data/"
  word_vec_dir=""
else
  echo "hostname not found."
fi
echo "==============================="
echo "[Project path]: $src_path"
echo "[Data directory]: $data_dir"
echo "[WordVec directory]: $word_vec_dir"
echo "==============================="

export PYTHONPATH=$src_path
cd "$src_path"
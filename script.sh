#!/bin/bash

set -o errexit
set -o pipefail

BASDIR=$PWD

CMD="$1"
ARG2="$2"
ARG3="$3"
ARG4="$4"


install_conda(){
  echo ""
  echo ""
  echo ""
  if [ -d /anaconda3 ]
  then
    echo "================================================================="
    echo "anaconda already installed"
    echo "================================================================="
  else
    echo "================================================================="
    echo "Downloading & Installing Anaconda"
    echo "================================================================="
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /anaconda3
  fi
  echo ""
  echo ""
  echo ""

}

conda_init(){
  echo ""
  echo ""
  echo ""
  if grep -q "conda initialize" ~/.bashrc; then
    echo "================================================================="
    echo "Conda initialize already available in .bashrc"
    echo "================================================================="

  echo "Sourcing .bashrc"
  source ~/.bashrc
  conda init
  fi
}

download_code(){
  git clone https://github.com/sumanshusamarora/mlflow-sklearn-classification.git
  cd mlflow-sklearn-classification
  echo "*********************************************************"
  echo "Please copy the data file named final.csv in data folder"
  echo "*********************************************************"
}

create_mlflow_environment(){
  conda env create -f environment.yml
  conda activate mlflow-sklearn
}

spin_up_gui(){
  if [ -z "$ARG3" ]
  then
    port=5000
  else
    port=$ARG3
  fi


  if [ -z "$ARG2" ]
  then
    echo "Please input mlruns-dir"
  else
    mlrunsdir=$ARG2
    echo "Using port ${port}"
    echo "mlflow server --backend-store-uri file://${mlrunsdir} --default-artifact-root file://${mlrunsdir} --host 0.0.0.0 --port ${port} &"
    mlflow server --backend-store-uri "file://${mlrunsdir}" --default-artifact-root "file://${mlrunsdir}" --host 0.0.0.0 --port "${port}" &
  fi

}

case $CMD in
  setmeup)
    install_conda
    conda_init
    #download_code
    create_mlflow_environment
    ;;

  spinupgui)
    spin_up_gui
    ;;

  --help)
    echo ""
    echo "Usage bash script.sh <setmeup|spinupgui|--help>"
    echo ""
    exit 2
    ;;

  *)
    echo ""
    echo "Usage bash script.sh <setmeup|spinupgui|--help>"
    echo ""
    exit 2
esac



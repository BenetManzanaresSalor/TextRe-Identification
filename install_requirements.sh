conda create -n TRI_env python=3.9 -y &&
conda activate TRI_env &&
conda install pytorch=2.2.1=py3.9_cuda11.8_cudnn8_0 -c pytorch -c nvidia -y &&
conda install cuda=11.6.1=0 -c nvidia -y &&
conda install transformers=4.39.3 -c huggingface -y &&
conda install accelerate=0.28.0=pyhd8ed1ab_0 -c huggingface -y &&
conda install spacy=3.5.0 -c spacy -y &&
conda install spacy-model-en_core_web_lg=3.5.0 -c spacy -y &&
conda install numpy=1.26.3=py39h055cbcc_0 -c numpy -y &&
conda install pandas=1.5.3=py39hf11a4ad_0 -c pandas -y &&
conda install tqdm=4.65.0=py39hd4e2768_0 -c conda-forge -y
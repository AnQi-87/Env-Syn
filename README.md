# Env-Syn: Learning Drug Synergy through Environment-Conditioned Feature Modulation

## Abstract
Motivation: Drug combinations are crucial for overcoming resistance in cancer therapy. Although deep learning has achieved strong performance in synergy prediction, existing models often treat cell-specific features and paired drugs as a static background and fail to capture how the specific cell-drug environment dynamically modulates drug representations, thereby hindering the modeling of environment-specific synergistic effects.
Results: We propose Env-Syn, a framework for modeling drug-drug-cell interactions through Environment-Conditioned Feature Modulation, it incorporates a Residual Feature-wise Linear Modulation (R-FiLM) module that performs precise affine transformations on drug representations conditioned on paired drugs and cellular environments. Benchmark evaluations show that Env-Syn consistently outperforms state-of-the-art methods. Notably, the model exhibits exceptional generalization resilience in rigorous inductive scenarios. It maintains high predictive accuracy for unseen drugs with AUROC and AUPRC exceeding 0.81 in the Leave-drug-out setting, and further demonstrates strong cross-dataset reliability by surpassing a recall of 0.7 on independent test set. Furthermore, among 15 novel predicted drug combinations, eight are directly supported by literature evidence. These results demonstrate that Env-Syn is an effective computational tool for drug synergy discovery.
![model](https://github.com/AnQi-87/Env-Syn/blob/main/Env-Syn.png)


## Environment
### create a new conda environment
- conda create -n Env-Syn python=3.9
- conda activate Env-Syn

### install
- conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
- conda install -c dglteam/label/th24_cu124 dgl
- pip install torchdata==0.6.1 --no-deps
- pip install numpy pandas matplotlib tqdm scikit-learn
- pip install torch_geometric
- pip install pyyaml
- pip install pydantic
- pip install pandas
- pip install scikit-learn

## Run
- Run the utils_test.py file first

`python utils_test.py`

- Run the image_train.py file then

`python image_train.py --use_image_fusion --use_cl --lambda_cl 0.2 --temperature 0.1 --base_temperature 0.075 --TRAIN_BATCH_SIZE 128 --TEST_BATCH_SIZE 128`

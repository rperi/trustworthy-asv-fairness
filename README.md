# trustworthy-asv-fairness
Improving fairness of speaker representations using adversarial and multi-task learning methods

Paper (Submitted to Elsevier Computer Speech and Language): To train or not to train adversarially: A study of bias mitigation strategies for speaker recognition https://arxiv.org/pdf/2203.09122.pdf

## Highlights
* Systematic evaluation of biases with respect to gender in speaker verification systems at multiple operating points.
* Amount of fairness achieved through training data balancing depends on operating region of speaker verification system.
* Adversarial and multi-task training based embedding transformation techniques improve the fairness of existing speaker verification systems.
* Utility is an important consideration in choosing appropriate bias mitigation strategies. Multi-task technique improves fairness and retains utility, adversarial technique improves fairness at the cost of reduced utility.

## Shared data, models
1. Pre-trained models for each of the methods described in paper: https://drive.google.com/drive/folders/1m_mv_klf3ZFAREuv0gct1SPV8KftxyTm?usp=sharing
2. Baseline embeddings (from FairVoice [[2]](#2)) for training and evaluation: https://drive.google.com/drive/folders/1Yb6GEultj4ig1kVb3h19Z7Mzz_UHU6Po?usp=sharing
3. Transformed embeddings (extracted using above pre-trained models) on eval-dev and eval-test datasets: https://drive.google.com/drive/folders/1HIB0Z7fEMFjOBXg_DrcXAZqUvuoDTyJl?usp=sharing
4. Verification trials based on Mozilla CommonVoice (MCV) and Voxceleb1-H audio: https://drive.google.com/drive/folders/1DJaGfuG6DaAFaQyICfE22rTR9aIhuif4?usp=sharing

## Conda environment
Will create a conda environment named uai_36 with required packages. Needed for subsequent steps
```
conda env create -f conda_env.yml
```

## Steps to run fairness evaluations
1. Download the transformed embeddings and trials from the above shared links, and place them in local directory of choice (<data_dir>)
2. Evaluate models with fairness metrics (Fairness Discrepancy Rate: [[1]](#1))

```
bash run_compute_auFDR.sh <data_dir> <test_split> <method>
```

<data_dir> is the chosen directory to save downloaded embeddings and trials files

<test_split> is one of "eval-dev", "eval-test" or "voxceleb1_h".

\<method\> is one of "AT", "MTL", "NLDR", "UAI", "UAI-AT" or "UAI-MTL"

For example, to run evaluations for "eval-dev" set and for "UAI-MTL" method

```
bash run_compute_auFDR.sh <data_dir> eval-dev UAI-MTL
```

## Steps to transform the baseline embeddings (Pre-transformed embeddings are already provided in the above links)
1. Download the baseline embeddings (FairVoice) and saved models from the above shared links, and place them in local directory of choice (<data_dir>)
2. Transform the baseline embeddings

```
bash run_transform.sh <data_dir> <test_split> <method>
```

<data_dir> is the chosen directory to save downloaded embeddings and trials files

<test_split> is one of "eval-dev", "eval-test" or "voxceleb1_h".

\<method\> is one of "AT", "MTL", "NLDR", "UAI", "UAI-AT" or "UAI-MTL"

For example, to transform embeddings for "eval-dev" set and for "UAI-MTL" method

```
bash run_transform.sh <data_dir> eval-dev UAI-MTL
```
 
## Methods
![Journal UAI_model_new](https://user-images.githubusercontent.com/23619674/155252585-42939d23-8486-4fe2-8f14-ae26176dacf8.png)

### UAI
![Eq1-3](https://user-images.githubusercontent.com/23619674/155253114-e298e144-9c25-491a-a34b-5adeec0296e5.png)

### UAI-AT AND UAI-MTL
![Equation_4](https://user-images.githubusercontent.com/23619674/155253115-93254052-ba5c-4819-acc9-e5f6e35cd6f0.png)

![modules](https://user-images.githubusercontent.com/23619674/155252784-48a106da-0681-4976-9e8b-a826c0e88474.png)

## Results
![KDE_IMPOSTOR](https://user-images.githubusercontent.com/23619674/159154211-1ba1219b-1b07-492e-9eba-72ecb9b3522c.png)
![KDE_target](https://user-images.githubusercontent.com/23619674/159154212-859122e6-9964-4196-ae22-823ce4ad858f.png)


## References
<a id="1">[1]</a> 
T. de Freitas Pereira, S. Marcel, Fairness in biometrics: a figure of merit to assess biometric verification systems

<a id="2">[2]</a>
G. Fenu, H. Lafhouli, M. Marras, Exploring algorithmic fairness in deep speaker verification, in: International Conference
on Computational Science and Its Applications, Springer, 2020, pp. 77–93.



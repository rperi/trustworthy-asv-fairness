# trustworthy-asv-fairness
Improving fairness of speaker representations using adversarial and collaborative learning methods
Paper: TO-BE-UPDATED

## Brief description
TO-DO

## Shared data, models
1. Pre-trained models for each of the methods described in paper: https://drive.google.com/drive/folders/1m_mv_klf3ZFAREuv0gct1SPV8KftxyTm?usp=sharing
2. Baseline embeddings (from FairVoice) for training and evaluation: https://drive.google.com/drive/folders/1Yb6GEultj4ig1kVb3h19Z7Mzz_UHU6Po?usp=sharing
3. Transformed embeddings (extracted using above pre-trained models) on eval-dev and eval-test datasets: https://drive.google.com/drive/folders/1HIB0Z7fEMFjOBXg_DrcXAZqUvuoDTyJl?usp=sharing
4. Verification trials based on Mozilla CommonVoice audio: https://drive.google.com/drive/folders/1DJaGfuG6DaAFaQyICfE22rTR9aIhuif4?usp=sharing

## Steps to run evaluations (on the eval-dev and eval-test datasets shown in paper)
1. Download the models, embeddings and trials from the above shared links, and place them in directory of choice (<data_dir>)
2. Evaluate models with fairness metrics

```
bash run_compute_auFDR.sh <data_dir> <test_split> <method>
```

<test_split> can be one of "dev" or "test". For example, to run evaluations for "eval-dev" set and for "UAI-MTL" method

```
bash run_compute_auFDR.sh <data_dir> dev UAI-MTL
```

## Methods
![Journal UAI_model_new](https://user-images.githubusercontent.com/23619674/155252585-42939d23-8486-4fe2-8f14-ae26176dacf8.png)

### UAI
![Eq1-3](https://user-images.githubusercontent.com/23619674/155253114-e298e144-9c25-491a-a34b-5adeec0296e5.png)

### UAI-AT AND UAI-MTL
![Equation_4](https://user-images.githubusercontent.com/23619674/155253115-93254052-ba5c-4819-acc9-e5f6e35cd6f0.png)

![modules](https://user-images.githubusercontent.com/23619674/155252784-48a106da-0681-4976-9e8b-a826c0e88474.png)

## Results



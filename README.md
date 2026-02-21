# base_deep_learning
Репозиторий для кратких решений основных задач из DL

## Audio
- [ASR](Audio/golos_asr_400.ipynb) - реализация [Quartznet](https://arxiv.org/pdf/1910.10261.pdf)
на датасете [Golos](https://github.com/salute-developers/golos)


## NLP
- [Word2Vec](NLP/word2vec.ipynb) - реализация модели Word2Vec в подходе Skip-Gram 
(из центрального слова предсказать контекст)
- [BiLSTM_CRF](NLP/BiLSTM_CRF.ipynb) - реализация сети со слоем 
[CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
(+ small optimization) для задачи NER (новости) + инференс на ONNX
- [Seq2Seq_with_Attention](NLP/seq2seq_with_attn.ipynb) - задача машинного перевода (EN->RU)
с механизмом внимания
- [Transformer](NLP/transformer.ipynb) - задача машинного перевода (EN->RU) с 
обучением трансформера

## CV
- [Style Transformer](CV/style_transformer.ipynb) - перенос стиля через обученную 
сверточную сеть из [репозитория](https://github.com/msuvorov7/real_styler)
- [SSD300 Detector](CV/SSD300_Detector.ipynb) - по мотивам семинаров Ozon Masters 
и [источника](https://www.kaggle.com/code/sdeagggg/ssd300-with-pytorch) SSD для 
детекции текста из [датасета](https://textvqa.org/textocr/dataset/)
- [MobileNet](CV/MobileNet.ipynb) - [V1](https://arxiv.org/pdf/1704.04861) 
и [V2](https://arxiv.org/pdf/1801.04381) для классификации
- [Darknet53](CV/Darknet53.ipynb) - [backbone](https://pjreddie.com/media/files/papers/YOLOv3.pdf) от yolov3 для классификации
- [SSDLite](CV/SSDLite.ipynb) - реализация с MobileNetV2 backbone на данных mscoco2017
- [YOLOv3](CV/YOLOv3.ipynb) - реализация [YOLOv3](https://arxiv.org/pdf/1804.02767) на данных mscoco2017

## Generative Models
- [PixelCNN](Generative%20Models/PixelCNN.ipynb) - реализация авторегрессионной 
генеративной модели PixelCNN на бинарном датасете MNIST
- [VAE](Generative%20Models/VAE.ipynb) - реализация VAE на датасете MNIST
- [RealNVP](Generative%20Models/RealNVP.ipynb) - реализация [RealNVP](https://arxiv.org/pdf/1605.08803)
на датасете MNIST
- [WGAN-GP](Generative%20Models/WGAN-GP.ipynb) - [WGAN with Gradient Penalty](https://arxiv.org/pdf/1704.00028)
- [WGAN-SN](Generative%20Models/WGAN-SN.ipynb) - [WGAN with Spectral Norm](https://arxiv.org/pdf/1802.05957)
- [DDPM](Generative%20Models/ddpm.ipynb) - реализация диффузионной модели ([DDPM](https://arxiv.org/pdf/2006.11239))
- [LDM](Generative%20Models/ldm.ipynb) - реализация модели латентной диффузии и VQVAE
- [DiT](Generative%20Models/DiT.ipynb) - реализация модели Diffusion Transformer (https://arxiv.org/pdf/2212.09748)

## RL
- [Cross Entropy Method discrete](RL/cross_entropy_taxi.py) - реализация метода Кросс-Энтропии
для среды Taxi-v3 с использованием сглаживания
- [Cross Entropy Method continuous](RL/cross_entropy_lunar_lander.py) - реализация метода
Кросс-Энтропии для среды LunarLander-v2
- [Model Free](RL/model_free.py) - реализация методов Q-Learning, Sarsa, Monte-Carlo для
среды CartPole-v1 с бинаризацией состояний среды
- [Deep Q-Networks](RL/deep_q_networks.py) - реализация методов DQN, HardTargetDQN, SoftTargetDQN,
DoubleDQN для среды LunarLander-v2
- [PPO Acrobot](RL/ppo_acrobot.py) - реализация метода PPO для среды Acrobot-v1
- [PPO Pendulum](RL/ppo_pendulum.py) - реализация метода PPO для среды Pendulum-v1
- [SAC Pendulum](RL/sac_pendulum.py) - реализация метода Soft Actor-Critic для среды Pendulum-v1
- [SAC CartPole](RL/sac_cartpole.py) - реализация метода Soft Actor-Critic для среды CartPole-v1
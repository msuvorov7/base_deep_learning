# base_deep_learning
Репозиторий для кратких решений основных задач из DL


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
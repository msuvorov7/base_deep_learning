# base_deep_learning
Репозиторий для кратких решений основных задач из DL


## NLP
- [Word2Vec](NLP/word2vec.ipynb) - реализация модели Word2Vec в подходе Skip-Gram 
(из центрального слова предсказать контекст)
- [BiLSTM_CRF](NLP/BiLSTM_CRF.ipynb) - реализация сети со слоем 
[CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
(+ small optimization) для задачи NER (новости) + инференс на ONNX

## CV
- [Style Transformer](CV/style_transformer.ipynb) - перенос стиля через обученную 
сверточную сеть из [репозитория](https://github.com/msuvorov7/real_styler)
- [SSD300 Detector](CV/SSD300_Detector.ipynb) - по мотивам семинаров Ozon Masters 
и [источника](https://www.kaggle.com/code/sdeagggg/ssd300-with-pytorch) SSD для 
детекции текста из [датасета](https://textvqa.org/textocr/dataset/)
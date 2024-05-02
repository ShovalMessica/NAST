## NAST: Noise Aware Speech Tokenization for Speech Language Models

Official implementation of NAST: Noise Aware Speech Tokenization for Speech Language Models. <br><br>
<p align="center">
  <img src="diagram.png" alt="diagram" style="width:55%;height:auto;"/>
</p>


<b>Abstract:</b> Speech tokenization is the task of representing speech signals as a sequence of discrete units. Such representations can be later used for various downstream tasks including automatic speech recognition, text-to-speech, etc. More relevant to this study, such representation serves as the basis of Speech Language Models. In this work, we tackle the task of speech tokenization under the noisy setup and present NAST: Noise Aware Speech Tokenization for Speech Language Models. NAST is composed of three main components: (i) a predictor; (ii) a residual encoder; and (iii) a decoder. We evaluate the efficiency of NAST considering several speech language modeling tasks, and show that NAST is superior to the evaluated baselines across all setups. Lastly, we analyze NAST and show its disentanglement properties and robustness to signal variations in the form of noise, reverberation, pitch-shift, and time-stretch. 

## Setup Environment
Create a conda environment, with python version 3.8 and install all the dependencies:
```python
def hello_world():
    print("Hello, world!")
```

## Usage

## Acoustic Model
For quantizing speech we learn NAST clustering over HuBERT Base acoustic representation. For using the pretrained model, please download from the link below.
- [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)

## Tokenization Model
You can download pretrained tokenization model from the list below. 
| NAST Model | Download Link |
|-----------------|-----------------|
| HuBERT Base + 50 units | [<center>download</center>](https://drive.google.com/file/d/1PDkV-m-kELx9fUeqmqPFWcbomHNddR1p/view?usp=drive_link)</center> |
| HuBERT Base + 100 units | <center>[download](https://drive.google.com/file/d/199YLQO8InNHfUbxkYjPLDwToPxmaiMi1/view?usp=drive_link)</center> |
| HuBERT Base + 200 units | <center>[download](https://drive.google.com/file/d/1KdyyYpWItsSJEoDLc-qo4YFGTCaUXmBQ/view?usp=drive_link)</center> |

## Unit Language Model (ULM)
You can download pretrained unit language models from the list below, or follow the [instructions](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/ulm) to train new models using fairseq. All language models were trained and evaluated on the deduplicated unit transcriptions of the respective NAST version.
| ULM Model | Download Link |
|-----------------|-----------------|
| NAST + 50 units | Row 1, Column 2 |
| NAST + 100 units | Row 2, Column 2 |
| NAST + 200 units | Row 3, Column 2 |

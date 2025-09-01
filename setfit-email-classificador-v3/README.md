---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Agende treinamento para a nova equipe.
- text: Parabéns pelo excelente trabalho na última apresentação!
- text: Desejo boa sorte no novo desafio.
- text: Parabéns por mais uma vitória.
- text: Preciso que você atualize o relatório financeiro até amanhã.
metrics:
- metric
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: neuralmind/bert-base-portuguese-cased
model-index:
- name: SetFit with neuralmind/bert-base-portuguese-cased
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: metric
      value: 1.0
      name: Metric
---

# SetFit with neuralmind/bert-base-portuguese-cased

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | <ul><li>'Prezada equipe de operações, identifiquei que o relatório de estoque consolidado deste mês apresenta divergências significativas em relação aos registros individuais de cada filial. Solicito que cada responsável revise os dados de sua unidade, destaque os itens que apresentam inconsistência e envie um relatório detalhado até quinta-feira. Esse levantamento é essencial para que possamos corrigir a contagem antes do fechamento financeiro mensal.'</li><li>'Boa tarde, notei que os acessos à pasta compartilhada de projetos foram alterados recentemente. Antes eu conseguia visualizar os documentos, mas agora não tenho permissão. Poderiam verificar se houve alguma mudança na política de acessos? Preciso consultar esses arquivos para concluir a apresentação que será enviada ao cliente até amanhã.'</li><li>'Revisar contrato enviado pelo setor jurídico.'</li></ul> |
| 0     | <ul><li>'Curiosidade do dia: descobri que nossa primeira versão pública saiu exatamente nesta mesma semana, anos atrás. É interessante olhar para trás e perceber o quanto evoluímos. Fica o registro histórico para quem gosta dessas memórias.'</li><li>'Obrigado por compartilhar o artigo interessante.'</li><li>'Boas festas a todos! Que este período traga paz, acolhimento e bons encontros. Que possamos voltar renovados para um novo ano de conquistas.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Evaluation

### Metrics
| Label   | Metric |
|:--------|:-------|
| **all** | 1.0    |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Parabéns por mais uma vitória.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 2   | 22.1833 | 92  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 148                   |
| 1     | 152                   |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0007 | 1    | 0.1658        | -               |
| 0.0333 | 50   | 0.1785        | -               |
| 0.0667 | 100  | 0.0369        | -               |
| 0.1    | 150  | 0.0015        | -               |
| 0.1333 | 200  | 0.0006        | -               |
| 0.1667 | 250  | 0.0004        | -               |
| 0.2    | 300  | 0.0003        | -               |
| 0.2333 | 350  | 0.0002        | -               |
| 0.2667 | 400  | 0.0002        | -               |
| 0.3    | 450  | 0.0002        | -               |
| 0.3333 | 500  | 0.0001        | -               |
| 0.3667 | 550  | 0.0001        | -               |
| 0.4    | 600  | 0.0001        | -               |
| 0.4333 | 650  | 0.0001        | -               |
| 0.4667 | 700  | 0.0001        | -               |
| 0.5    | 750  | 0.0001        | -               |
| 0.5333 | 800  | 0.0001        | -               |
| 0.5667 | 850  | 0.0001        | -               |
| 0.6    | 900  | 0.0001        | -               |
| 0.6333 | 950  | 0.0001        | -               |
| 0.6667 | 1000 | 0.0001        | -               |
| 0.7    | 1050 | 0.0001        | -               |
| 0.7333 | 1100 | 0.0001        | -               |
| 0.7667 | 1150 | 0.0001        | -               |
| 0.8    | 1200 | 0.0001        | -               |
| 0.8333 | 1250 | 0.0001        | -               |
| 0.8667 | 1300 | 0.0001        | -               |
| 0.9    | 1350 | 0.0001        | -               |
| 0.9333 | 1400 | 0.0001        | -               |
| 0.9667 | 1450 | 0.0001        | -               |
| 1.0    | 1500 | 0.0001        | -               |

### Framework Versions
- Python: 3.13.1
- SetFit: 1.1.3
- Sentence Transformers: 5.1.0
- Transformers: 4.56.0
- PyTorch: 2.8.0+cpu
- Datasets: 4.0.0
- Tokenizers: 0.22.0

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
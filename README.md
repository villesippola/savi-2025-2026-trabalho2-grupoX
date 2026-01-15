# Practical Work 2 - SAVI
==============

Group X

2025-2026

# Practical Work: SAVI-Mnist

**Classification and Detection of Handwritten Digits using Convolutional Neural Networks**

## Methodology

While the first work focused on classical computer vision methods (geometry and registration), this second work focuses on deep learning . The goal is to evolve from a simple classification problem (classic MNIST) to a more realistic and complex scenario: the detection and classification of multiple objects in larger images.

Students will consolidate their acquired knowledge of PyTorch (Convolutional Neural Network) architectures , CNN , evaluation metrics, and object detection techniques. The work evolves incrementally from the optimization of a classifier to the implementation of a complete object detector.

## Configuration and Prerequisites

Make sure you have the following libraries installed in your Python environment (in addition to those already used in TP1):
*   `torch` and `torchvision` (PyTorch)
*   `scikit-learn` (for calculating advanced metrics)
*   `tqdm` (for progress bars)
*   `seaborn` (for visualization of confusion matrices)
*   `git` (to clone the dataset generation repository)

**Base Data:** The MNIST dataset will be downloaded automatically via `torchvision` for Task 1. For subsequent tasks, a synthetic dataset will be generated. 

## Tasks

---

### Task 1: Optimized CNN Classifier (Full MNIST)

To classify individual MNIST digits, we developed a custom Convolutional Neural Network (CNN) using PyTorch.

* **Architecture:** The model (`ModelBetterCNN`) improves upon the baseline by increasing network depth and adding regularization mechanisms.
    * **Input:** 28x28 Grayscale images.
    * **Layers:** We utilized 3 Convolutional blocks. Each block consists of:
        * `Conv2d`: Feature extraction.
        * `BatchNorm2d`: To stabilize training and allow higher learning rates.
        * `ReLU`: Non-linear activation.
        * `MaxPool2d`: To reduce spatial dimensions and computation.
    * **Regularization:** `Dropout` layers were added before the fully connected layers to prevent overfitting.
    * **Loss Function:** `MSELoss` (with One-Hot Encoded labels) / `CrossEntropyLoss`.

| Digit | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| 0 | 0.9910 | 0.9980 | 0.9944 |
| 1 | 0.9947 | 0.9965 | 0.9956 |
| 2 | 0.9942 | 0.9913 | 0.9927 |
| 3 | 0.9950 | 0.9941 | 0.9946 |
| 4 | 0.9869 | 0.9939 | 0.9904 |
| 5 | 0.9922 | 0.9922 | 0.9922 |
| 6 | 0.9906 | 0.9916 | 0.9911 |
| 7 | 0.9913 | 0.9951 | 0.9932 |
| 8 | 0.9918 | 0.9938 | 0.9928 |
| 9 | 0.9939 | 0.9752 | 0.9845 |
| Overall | 0.9921 | 0.9922 | 0.9921 |

Overall accuracy: 0.9922

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/eec9b919-ca89-4c82-9974-df6ec4f9b3e3" />

<img width="640" height="480" alt="training" src="https://github.com/user-attachments/assets/e8731878-76b8-476a-b051-cac6405b70d1" />


---

### Task 2: Synthetic Dataset Generation

We generated a "Scene" dataset to simulate object detection tasks. The generation script (`generate_data.py`) places MNIST digits onto a larger canvas (128x128) while preventing overlap.

* **Variability:** We created 4 dataset versions to test robustness:
    * **Type A:** 1 Digit, Fixed Scale (28x28).
    * **Type B:** 1 Digit, Random Scale (22x22 to 36x36).
    * **Type C:** 3-5 Digits, Fixed Scale.
    * **Type D:** 3-5 Digits, Random Scale.

---

### Task 3: Sliding Window Detection

**Objetivo:** Utilizar o classificador treinado na Tarefa 1 para encontrar dígitos nas "cenas" da Tarefa 2, sem re-treinar a rede.

1.  **Abordagem:** Implemente uma técnica de *Sliding Window* (Janela Deslizante).
    *   Percorra a imagem de entrada (do dataset da Tarefa 2) com janelas de tamanho 28x28 (ou redimensionadas).
    *   Passe cada recorte (crop) pela rede treinada na Tarefa 1.
2.  **Thresholding:** Defina um limiar de confiança (baseado na saída *softmax*) para decidir se um recorte contém um dígito ou é fundo (background).
    *   *Nota:* Como a rede da Tarefa 1 nunca viu "fundo", ela tentará classificar tudo como um dígito. Terá de lidar com este problema (e.g., analisando a entropia da saída ou a magnitude dos logits).
3.  **Visualização:** Desenhe as caixas delimitadoras (bounding boxes) onde a rede detetou dígitos com alta confiança sobre a imagem original.
4.  **Avaliação Qualitativa:** Discuta no README a eficiência desta abordagem (tempo de execução) e os problemas encontrados (falsos positivos, precisão da localização).

**Deliverable:** Código Python **main_sliding_window.py**. Inclua exemplos de imagens com as deteções no README.

---

### Task 4: Integrated Detector and Classifier

**Objetivo:** Alterar a arquitetura ou a estratégia de treino para realizar a deteção e classificação de forma mais eficiente e precisa.

1.  **Nova Abordagem:** Desenvolva uma solução que supere as limitações da janela deslizante. Algumas sugestões:
    *   **Conversão para FCN:** Converta as camadas `Linear` (fully connected) da sua CNN em camadas Convolucionais (Fully Convolutional Network). Isso permite passar a imagem inteira de uma vez e obter um mapa de calor de ativações.
    *   **Regressão de Bounding Box:** Altere a saída da rede para prever também as coordenadas `(x, y, w, h)` além da classe (abordagem simplificada tipo YOLO/R-CNN).
    *   **Region Proposals (RPN)**: Implemente um mecanismo de "Propostas de Região". Pode ser uma sub-rede dedicada (Region Proposal Network) que aprende a sugerir onde existem objetos antes de classificar (abordagem Two-Stage similar à Faster R-CNN), ou utilizar algoritmos rápidos de segmentação para gerar candidatos.
    *   **Re-treino:** Utilize o dataset da Tarefa 2 para treinar esta nova rede, permitindo que ela aprenda a distinguir "fundo" de "dígito".
2.  **Implementação:** Crie o treino e a inferência para esta nova arquitetura.
3.  **Comparação:** Compare os resultados (visuais e, se possível, de métricas) com a abordagem da Tarefa 3. A nova abordagem é mais rápida? É mais precisa?

**Deliverable:** Código Python **main_improved_detection.py**. Relatório detalhando as alterações arquiteturais feitas.

---

## Entrega

Para cada tarefa, deverá submeter:
*   O código Python (`.py`) claro, comentado e funcional.
*   A entrega é feita com um repositório chamado `savi-2025-2026-trabalho2-grupoX`, em que X é o número do grupo. 
*   O `README.md` deve ser o relatório principal, contendo:
    *   **Metodologia:** Explicação das arquiteturas de rede escolhidas (desenhos/diagramas são valorizados).
    *   **Resultados T1:** Matrizes de confusão e tabela de métricas (F1, Precision, Recall).
    *   **Análise de Dados T2:** Estatísticas e exemplos do dataset gerado.
    *   **Deteção T3 vs T4:** Comparação visual e discussão sobre performance (tempo vs qualidade) entre a janela deslizante e a abordagem melhorada.
    *   **Dificuldades:** Descrição dos principais desafios e soluções encontradas.

## Dicas e Sugestões

*   **GPU:** O treino com o dataset completo e a geração de dados podem ser pesados. Use a GPU (CUDA) se disponível. Verifique sempre com `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
*   **Overfitting:** Se a accuracy de treino for muito superior à de teste na Tarefa 1, o modelo está em *overfitting*. Aumente o Dropout ou reduza a complexidade do modelo.

# Fraud Detection using AI

A curated list of papers concerning fraud detection using AI

### Survey

**July, 2023**: We summarized the current state of the fraud detection using transformer and neural network models in the following survey paper (work in progress)

> `<Author1>`, `<Author2>`.  **Fraud Detection using Artificial Intelligence: A Survey**.

### Abstract

The rapid advancement of artificial intelligence (AI) has significantly impacted various sectors, with fraud detection being one of the most critical applications. Leveraging transformer models and neural networks, AI has demonstrated remarkable capabilities in identifying and preventing fraudulent activities with greater accuracy and efficiency. This survey aims to provide a comprehensive summary of the most influential research papers that have explored the use of AI in fraud detection. By analyzing various methodologies, model architectures, and their respective outcomes, this survey highlights the key trends and innovations in the field. Looking forward, the future direction of AI in fraud detection is poised to focus on data compression and downstream tasks. Effective data compression techniques will be essential in handling the vast amounts of data generated and ensuring that models can process this information efficiently. Furthermore, the integration of AI-driven fraud detection with downstream tasks, such as real-time decision-making and automated response systems, will enhance the overall effectiveness and responsiveness of fraud prevention strategies. By advancing in these areas, AI can continue to evolve, providing more sophisticated and agile solutions to the ever-growing challenge of fraud detection.

### Citation

## Related surveys

* [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection](https://doi.org/10.48550/arXiv.2303.18138)
  
  <details>
  <summary> Sihao Hu et al.  
      <em>The ACM Web Conference</em>, 2023 </summary>
    As various forms of fraud proliferate on Ethereum, it is imperative to safeguard against these malicious activities to protect susceptible users from being victimized. While current studies solely rely on graph-based fraud detection approaches, it is argued that they may not be well-suited for dealing with highly repetitive, skew-distributed and heterogeneous Ethereum transactions. To address these challenges, we propose BERT4ETH, a universal pre-trained Transformer encoder that serves as an account representation extractor for detecting various fraud behaviors on Ethereum. BERT4ETH features the superior modeling capability of Transformer to capture the dynamic sequential patterns inherent in Ethereum transactions, and addresses the challenges of pre-training a BERT model for Ethereum with three practical and effective strategies, namely repetitiveness reduction, skew alleviation and heterogeneity modeling. Our empirical evaluation demonstrates that BERT4ETH outperforms state-of-the-art methods with significant enhancements in terms of the phishing account detection and de-anonymization tasks. The code for BERT4ETH is available at: <a href="https://github.com/git-disl/BERT4ETH" rel="external noopener nofollow" class="link-external link-https">this https URL</a>.
  </details>
   <p align="center"><a href="https://doi.org/10.48550/arXiv.2303.18138"><img height='500' src="figs/model_bert4eth.png"></a></p>
* [Generative Pretraining at Scale: Transformer-Based Encoding of Transactional Behavior for Fraud Detection](https://doi.org/10.48550/arXiv.2312.14406)
  
  <details>
  <summary> Ze Yu Zhao et al. 
      <em>arXiv preprint arXiv:2312.14406</em>, 2023 </summary>
    In this work, we introduce an innovative autoregressive model leveraging Generative Pretrained Transformer (GPT) architectures, tailored for fraud detection in payment systems. Our approach innovatively confronts token explosion and reconstructs behavioral sequences, providing a nuanced understanding of transactional behavior through temporal and contextual analysis. Utilizing unsupervised pretraining, our model excels in feature representation without the need for labeled data. Additionally, we integrate a differential convolutional approach to enhance anomaly detection, bolstering the security and efficacy of one of the largest online payment merchants in China. The scalability and adaptability of our model promise broad applicability in various transactional contexts.
  </details>

* [AUC-oriented Graph Neural Network for Fraud Detection](https://doi.org/10.1145/3485447.3512178)
  
  <details>
  <summary> Mengda Huang et al. 
      <em>WWW '22: Proceedings of the ACM Web Conference</em>, 2022 </summary>
    Though Graph Neural Networks (GNNs) have been successful for fraud detection tasks, they suffer from imbalanced labels due to limited fraud compared to the overall userbase. This paper attempts to resolve this label-imbalance problem for GNNs by maximizing the AUC (Area Under ROC Curve) metric since it is unbiased with label distribution. However, maximizing AUC on GNN for fraud detection tasks is intractable due to the potential polluted topological structure caused by intentional noisy edges generated by fraudsters. To alleviate this problem, we propose to decouple the AUC maximization process on GNN into a classifier parameter searching and an edge pruning policy searching, respectively. We propose a model named AO-GNN (Short for AUC-oriented GNN), to achieve AUC maximization on GNN under the aforementioned framework. In the proposed model, an AUC-oriented stochastic gradient is applied for classifier parameter searching, and an AUC-oriented reinforcement learning module supervised by a surrogate reward of AUC is devised for edge pruning policy searching. Experiments on three real-world datasets demonstrate that the proposed AO-GNN patently outperforms state-of-the-art baselines in not only AUC but also other general metrics, e.g. F1-macro, G-means.
  </details>

* [A Neural Network Ensemble With Feature Engineering for Improved Credit Card Fraud Detection](https://doi.org/10.1109/ACCESS.2022.3148298)
  
  <details>
  <summary> E. Esenogho et al. 
      <em>IEEE Access, vol. 10, pp. 16400-16407</em>, 2022 </summary>
    Recent advancements in electronic commerce and communication systems have significantly increased the use of credit cards for both online and regular transactions. However, there has been a steady rise in fraudulent credit card transactions, costing financial companies huge losses every year. The development of effective fraud detection algorithms is vital in minimizing these losses, but it is challenging because most credit card datasets are highly imbalanced. Also, using conventional machine learning algorithms for credit card fraud detection is inefficient due to their design, which involves a static mapping of the input vector to output vectors. Therefore, they cannot adapt to the dynamic shopping behavior of credit card clients. This paper proposes an efficient approach to detect credit card fraud using a neural network ensemble classifier and a hybrid data resampling method. The ensemble classifier is obtained using a long short-term memory (LSTM) neural network as the base learner in the adaptive boosting (AdaBoost) technique. Meanwhile, the hybrid resampling is achieved using the synthetic minority oversampling technique and edited nearest neighbor (SMOTE-ENN) method. The effectiveness of the proposed method is demonstrated using publicly available real-world credit card transaction datasets. The performance of the proposed approach is benchmarked against the following algorithms: support vector machine (SVM), multilayer perceptron (MLP), decision tree, traditional AdaBoost, and LSTM. The experimental results show that the classifiers performed better when trained with the resampled data, and the proposed LSTM ensemble outperformed the other algorithms by obtaining a sensitivity and specificity of 0.996 and 0.998, respectively.
  </details>

* [Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms](https://doi.org/10.1109/ACCESS.2022.3166891)
  
  <details>
  <summary> F. K. Alarfaj et al. 
      <em>IEEE Access, vol. 10, pp. 16400-16407</em>, 2022 </summary>
    People can use credit cards for online transactions as it provides an efficient and easy-to-use facility. With the increase in usage of credit cards, the capacity of credit card misuse has also enhanced. Credit card frauds cause significant financial losses for both credit card holders and financial companies. In this research study, the main aim is to detect such frauds, including the accessibility of public data, high-class imbalance data, the changes in fraud nature, and high rates of false alarm. The relevant literature presents many machines learning based approaches for credit card detection, such as Extreme Learning Method, Decision Tree, Random Forest, Support Vector Machine, Logistic Regression and XG Boost. However, due to low accuracy, there is still a need to apply state of the art deep learning algorithms to reduce fraud losses. The main focus has been to apply the recent development of deep learning algorithms for this purpose. Comparative analysis of both machine learning and deep learning algorithms was performed to find efficient outcomes. The detailed empirical analysis is carried out using the European card benchmark dataset for fraud detection. A machine learning algorithm was first applied to the dataset, which improved the accuracy of detection of the frauds to some extent. Later, three architectures based on a convolutional neural network are applied to improve fraud detection performance. Further addition of layers further increased the accuracy of detection. A comprehensive empirical analysis has been carried out by applying variations in the number of hidden layers, epochs and applying the latest models. The evaluation of research work shows the improved results achieved, such as accuracy, f1-score, precision and AUC Curves having optimized values of 99.9%,85.71%,93%, and 98%, respectively. The proposed model outperforms the state-of-the-art machine learning and deep learning algorithms for credit card detection problems. In addition, we have performed experiments by balancing the data and applying deep learning algorithms to minimize the false negative rate. The proposed approaches can be implemented effectively for the real-world detection of credit card fraud.
  </details>


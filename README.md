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
* [An edge feature aware heterogeneous graph neural network model to support tax evasion detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417422019212)
  <details>
  <summary>
  Bin Shi et.al. <em>Expert Systems With Applications,2023 </em>
  </summary>
  Tax evasion is an illegal activity that causes severe losses of government revenues and disturbs the economic order. To alleviate this problem, decision support systems that enable tax authorities to detect tax evasion efficiently have been proposed. Recent researches tend to use graph to model the tax scenario and leverage graph mining techniques to conduct tax evasion detection, as so to make full use of the rich interactive information between taxpayers and improve the detection performance. However, a more favorable graph mining solution, graph neural networks, has not yet been thoroughly investigated in such settings, leaving space for further improvement. Therefore, in this paper, we propose a novel graph neural network model, named Eagle, to detect tax evasion under the heterogeneous graph. Specifically, based on the guidance of our designed metapaths, Eagle can extract more comprehensive features through a hierarchical attention mechanism that fully aggregates taxpayers’ features with their relations. We evaluate Eagle on real-world tax dataset. The extensive experimental results show that our model performs 15.71% better than state-of-the-art tax evasion detection methods in the classification scenario, while improves 5.22% in the anomaly detection scenario.
  </details>

* [RR-PU: A Synergistic Two-Stage Positive and Unlabeled Learning Framework for Robust Tax Evasion Detection](https://ojs.aaai.org/index.php/AAAI/article/view/28665)
  <details>
  <summary>
  Shuzhi Cao, <em>The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)</em>, 2024
  </summary>
  Tax evasion, an unlawful practice in which taxpayers deliber- ately conceal information to avoid paying tax liabilities, poses significant challenges for tax authorities. Effective tax eva- sion detection is critical for assisting tax authorities in mit- igating tax revenue loss. Recently, machine-learning-based methods, particularly those employing positive and unlabeled (PU) learning, have been adopted for tax evasion detection, achieving notable success. However, these methods exhibit two major practical limitations. First, their success heavily relies on the strong assumption that the label frequency (the fraction of identified taxpayers among tax evaders) is known in advance. Second, although some methods attempt to es- timate label frequency using approaches like Mixture Pro- portion Estimation (MPE) without making any assumptions, they subsequently construct a classifier based on the error- prone label frequency obtained from the previous estimation. This two-stage approach may not be optimal, as it neglects error accumulation in classifier training resulting from the es- timation bias in the first stage. To address these limitations, we propose a novel PU learning-based tax evasion detection framework called RR-PU, which can revise the bias in a two- stage synergistic manner. Specifically, RR-PU refines the la- bel frequency initialization by leveraging a regrouping tech- nique to fortify the MPE perspective. Subsequently, we in- tegrate a trainable slack variable to fine-tune the initial label frequency, concurrently optimizing this variable and the clas- sifier to eliminate latent bias in the initial stage. Experimen- tal results on three real-world tax datasets demonstrate that RR-PU outperforms state-of-the-art methods in tax evasion detection tasks.
  </details>

* [A Survey of Tax Risk Detection Using Data Mining Techniques](https://www.sciencedirect.com/science/article/pii/S2095809923003867)
  
  <details>
  <summary>
  Qinghua Zheng, <em>Engineering Volume 34,</em> 2024
  </summary>
  Tax risk behavior causes serious loss of fiscal revenue, damages the country’s public infrastructure, and disturbs the market economic order of fair competition. In recent years, tax risk detection, driven by information technology such as data mining and artificial intelligence, has received extensive attention. To promote the high-quality development of tax risk detection methods, this paper provides the first comprehensive overview and summary of existing tax risk detection methods worldwide. More specifically, it first discusses the causes and negative impacts of tax risk behaviors, along with the development of tax risk detection. It then focuses on data-mining-based tax risk detection methods utilized around the world. Based on the different principles employed by the algorithms, existing risk detection methods can be divided into two categories: relationship-based and non-relationship-based. A total of 14 risk detection methods are identified, and each method is thoroughly explored and analyzed. Finally, four major technical bottlenecks of current data-driven tax risk detection methods are analyzed and discussed, including the difficulty of integrating and using fiscal and tax fragmented knowledge, unexplainable risk detection results, the high cost of risk detection algorithms, and the reliance of existing algorithms on labeled information. After investigating these issues, it is concluded that knowledge-guided and data-driven big data knowledge engineering will be the development trend in the field of tax risk in the future; that is, the gradual transition of tax risk detection from informatization to intelligence is the future development direction.
  </details>

* [Explainable Fraud Detection with Deep Symbolic Classification](https://arxiv.org/abs/2312.00586)
  
  <details>
  <summary>
  Samantha Visbeek et.al. ,<em> XAIFIN’23, 4th ACM International Conference on AI in Finance (ICAIF) </em> 2023
  </summary>
  There is a growing demand for explainable, transparent, and data-
  driven models within the domain of fraud detection. Decisions made
  by the fraud detection model need to be explainable in the event of
  a customer dispute. Additionally, the decision-making process in
  the model must be transparent to win the trust of regulators, ana-
  lysts, and business stakeholders. At the same time, fraud detection
  solutions can benefit from data due to the noisy and dynamic nature
  of fraud detection and the availability of large historical data sets.
  Finally, fraud detection is notorious for its class imbalance: there
  are typically several orders of magnitude more legitimate transac-
  tions than fraudulent ones. In this paper, we present Deep Symbolic
  Classification (DSC), an extension of the Deep Symbolic Regression
  framework to classification problems. DSC casts classification as a
  search problem in the space of all analytic functions composed of a
  vocabulary of variables, constants, and operations and optimizes
  for an arbitrary evaluation metric directly. The search is guided by a
  deep neural network trained with reinforcement learning. Because
  the functions are mathematical expressions that are in closed-form
  and concise, the model is inherently explainable both at the level of
  a single classification decision and at the model’s decision process
  level. Furthermore, the class imbalance problem is successfully ad-
  dressed by optimizing for metrics that are robust to class imbalance
  such as the F1 score. This eliminates the need for problematic over-
  sampling and undersampling techniques that plague traditional
  approaches. Finally, the model allows to explicitly balance between
  the prediction accuracy and the explainability. An evaluation on the
  PaySim data set demonstrates competitive predictive performance
  with state-of-the-art models, while surpassing them in terms of
  explainability. This establishes DSC as a promising model for fraud
  detection systems.
  </details>


* [xFraud: Explainable Fraud Transaction Detection](https://arxiv.org/pdf/2011.12193)
  <details>
  <summary>
  Susie Xi Rao et.al., <em>48th International Conference on Very Large Databases</em>,2022.
  </summary>
  At online retail platforms, it is crucial to actively detect the risks of transactions to improve customer experience and minimize financial loss. In this work, we propose xFraud, an explainable fraud transaction prediction framework which is mainly composed of a detector and an explainer. The xFraud detector can effectively and efficiently predict the legitimacy of incoming transactions. Specifically, it utilizes a heterogeneous graph neural network to learn expressive representations from the informative heterogeneously typed entities in the transaction logs. The explainer in xFraud can generate meaningful and human-understandable explanations from graphs to facilitate further processes in the business unit. In our experiments with xFraud on real transaction networks with up to 1.1 billion nodes and 3.7 billion edges, xFraud is able to outperform various baseline models in many evaluation metrics while remaining scalable in distributed settings. In addition, we show that xFraud explainer can generate reasonable explanations to significantly assist the business analysis via both quantitative and qualitative evaluations.
  </details>
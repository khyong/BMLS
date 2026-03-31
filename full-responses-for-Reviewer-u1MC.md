

---
>W1 & W6. The SDP relaxation is not well justified
* (line 122-140 L) Through improved writing, we have defined the convex program and Eq. 11 (Eq. 4 in the revised paper) prior to Lemma 4.1.

---
>W2. Direct empirical result to prove or explain when the NC occur.
* We appreciate the reviewer's insightful comments. Due to the charater limit, please refer to the response to Reviewer qTac (W7 & Q3)
* Q1. (Average of Test Accuracy) The average test accuracy refers to the mean of test accuracies obtained across different random seeds. The performance of all experiments is measured using Top-1 test accuracy, which is the standard metric used in conventional long-tailed benchmarks, and further details regarding the evaluation criteria can be found in (line 268-272 R) and Appendix F.

---
>W3 & Q1. Average of Test Accuracy
* The average test accuracy refers to the mean of test accuracies obtained across different random seeds. The performance of all experiments is measured using Top-1 test accuracy, which is the standard metric used in conventional long-tailed benchmarks, and further details regarding the evaluation criteria can be found in (line 268-272 R) and Appendix F.

---
>W4. Reproducibility on class-balanced sampler
* (Table 4) There was a misunderstanding regarding the notation. All methods except those marked with $\dagger$ are reproduced under the same experimental settings as our proposed method, and the background shading was used solely to distinguish the diff rows (i.e., the difference between the baseline and our proposed method). We have revised the presentation to make this clearer.

---
>W5 & Q3, L2, L3. Scalability - mini-batch size
* Following the reviewer's suggestion, we compared the performance of three methods, Mixup, BMLS, and BMLS+MS, on CIFAR10/100-LT (imbalance factor: 100) with mini-batch sizes set to [16, 32, 64, 128, 256, 512] using 5 random seeds. The results show that while performance is affected by the mini-batch size, BMLS and BMLS+MS consistently outperform Mixup across all mini-batch sizes on CIFAR100-LT, except when bsz is 16. In particular, the performance on the Few category is notably higher at a batch size of 16, which suggests that our proposed method further alleviates minority collapse.

---
>W6. Paper Structure
*  We appreciate the reviewer's insightful comments. Please refer to the response to Reviewer 8hso (W10 & Q1.)

---
>W7. Reference for imbalanced learning
* (line 37-40 L) We have incorporated additional relevant references, into the revised paper.

---
>W8. The tables and figures are not well described
* (line 268-272 R) For the sake of conciseness, rather than indicating test accuracy in every table, we have noted at the beginning of the experimental section that all tables report test accuracy.

---
>W9. Appendix Overreliance
* Please refer to the response to Reviewer 8hso (W11 & Q3.)

---
>Q2 & Q4, L1, L2. Scalability

The most ideal condition for our proposed method is when the dataset size is greater than the square of the number of classes $(N \geq K^2)$. Under this condition, the imbalance can be completely resolved by generating an equal number of mixed samples for all mixed labels, and its effectiveness has been demonstrated on CIFAR10/100-LT, which satisfies this condition. However, as the reviewer pointed out, when the number of classes increases $(N < K^2)$, the balance of mixed labels cannot be fully guaranteed by our proposed method. Nevertheless, even under such conditions, our method still demonstrates competitive performance compared to other class-balanced samplers, as demonstrated by the results in Table 6 (Table 7 in the revised paper) in Appendix A. In Places-LT, both BMLS and MS achieve higher performance than other class-balanced samplers, and when BMLS alone is employed, it attains the highest performance on ImageNet-LT and competitive performance on iNaturalist2018, the most challenging benchmark considered in this work. Furthermore, the scaling experiment (Table 7) (Table 8 in the revised paper) demonstrates that the proposed method can be effectively handled through straightforward engineering adjustments.

This study identifies a novel research problem (does mixed label imbalance matter to minority collapse in imbalanced learning?), theoretically proves it (mixed label imbalance causes minority collapse), and proposes BMLS to address it along with the MS classifier to further reinforce it (balancing mixed labels mitigates minority collapse). The primary contribution lies in demonstrating that resolving this problem is a critical factor in improving model performance in empirical long-tailed settings. While we acknowledge the scalability issue in large-class settings, Appendix A demonstrates that our works retains its effectiveness to a reasonable degree in large-class settings (Places-LT, ImageNet-LT), and the scaling experiment on ImageNet-LT highlights its potential for further development. We hope that this additional effort, which establishes a foundation for future investigation, will be taken into consideration.

---

>L3. What is the effect on the training time/compute and parametrization - should use larger mini batch size?
* As a result of below tables, our proposed methods increase the training time about x1.30 in CIFAR10-LT and x1.09 in CIFAR100-LT. Other factors such as Peak GPU memory usage and the number of parameters are no significant difference.

CIFAR10-LT

| Method  | Time (s/epoch) | Peak GPU Mem (MB) | # params |
| ------- | -------------- | ----------------- | -------- |
| Mixup   | 5.2023         | 338.79            | 464,154  |
| BMLS    | 6.6693         | 338.79            | 464,154  |
| BMLS+MS | 6.7795         | 338.81            | 464,154  |

CIFAR100-LT

| Method  | Time (s/epoch) | Peak GPU Mem (MB) | # params  |
| ------- | -------------- | ----------------- | --------- |
| Mixup   | 5.7773         | 670.82            | 1,861,508 |
| BMLS    | 6.1372         | 670.82            | 1,861,508 |
| BMLS+MS | 6.2828         | 670.90            | 1,861,508 |


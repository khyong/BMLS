>W1 & Q1. Strong assumptions on dimensionality

(*The strong assumption guarantees the theoretical robustness*) The conditions $p \geq K^2+K$ in Lemma 4.1 and Proposition 4.3 and $p \geq 2K^2$ in Theorem 5.1 are imposed to ensure the existence of the left-hand side terms in Equation 5 within the SDP, and thus bear no direct relation to the onset of minority collapse. In fact, in Theorem 4.2, this constraint is relaxed to $p \geq K$. Therefore, our key proposition that mixed label imbalance plays a critical role in minority collapse under Mixup in imbalanced learning remains valid even under practical condition

(*Practical realism should still be taken into account w.r.t BMLS and MS*) However, since the claim in this paper that balancing mixed labels mitigates minority collapse stems from Proposition 4.3 and Theorem 5.1, accounting for practical realism is also warranted. Nevertheless, the conditions $p \geq K^2+K$ and $p \geq 2K^2$ are seldom satisfied in datasets beyond those with few classes, such as CIFAR10-LT. In practice, due to these limitations, although balancing all mixed labels substantially alleviates minority collapse relative to other baselines, the accuracy of the *few* category reamins inferior to that of other categories.

(*BMLS and MS empirically demonstrate their effectiveness*) Nevertheless, through the empirical validation of BMLS and MS across CIFAR100-LT, Places-LT, and ImageNet-LT, none of which fully satisfy our theoretical assumptions, we empirically substantiate the practical efficacy of our theoretical findings in resolving minority collapse in imbalanced learning.

In summary, the strong constraints employed in the theoretical analysis of this paper are imposed to rigorously consolidate the theoretical proof of the problem under investigation. Even though these constraints may not be satisfied in practice, the balancing sampler derived from the theory has yielded sufficiently promising results, constituting a meaningful contribution of our findings. Just as mini-batch SGD is employed to address the practical limitations of full-batch gradient descent, and just as GNC theoretically extended the constraint $D \geq K-1$ of NC to $D < K-1$ to establish uniform distribution on the hypersphere, the development of theorems and methods for balancing mixed labels is expected to play a pivotal role from both theoretical and empirical perspectives in resolving minority collapse in imbalanced learning. This paper lays the theoretical groundwork as a starting point and demonstrates its potential, thereby making a meaningful contribution.

****

> W2. Theoretical support for the Mixed-Singleton classifier

We have provided a proof for Theorem 4.5 (Theorem 5.1 in the revised paper). Through this, we believe that the theoretical support for the claim that BMLS still mitigates minority collapse in LPM$_{\lambda}$​-MS, which is the LPM with Mixup incorporating the MS classifier, has been strengthened. In other words, to summarize, this part was inspired by the observation that minority collapse occurs regardless of the Mixup ratio, which motivated the proposal of the Mixed-Singleton classifier. Therefore, this theorem can be understood as the process of theoretically verifying that BMLS, when used in conjunction with MS classifier, can still mitigate minority collapse. A more comprehensive theoretical analysis on MS classifier is left as future work, through which we hope to address the scalability issue.

---

> W3 & Q2. $\lambda$-independence

To our knowledge, $\lambda$-independence stems from the characteristic that last-layer features are linearly interpolated when Mixup is applied to LPM or UFM, and thus, as the reviewer pointed out, this may be regarded as an artifact of proxy models. However, according to [A1], although this does not perfectly conform to the theoretical framework in real deep networks (WideResNet, ViT) where the role of $\lambda$ may become more complex, the last-layer features of mixed samples exhibit a tendency to be positioned between the two classes used in Mixup, following a linear interpolation scheme. Therefore, it is reasonable to posit that $\lambda$-independence may also manifest to a certain degree in real deep networks beyond proxy models. This suggests that the proposed Mixed-Singleton classifier can serve as a method applicable in general settings. At the same time, rather than computing the loss of mixed samples as a simple sum of scalar predictions for the classes engaged in the mixed label, the Mixed-Singleton classifier, which treats the mixed label as a singleton class and derives the loss therefrom, has the effect of potentially reducing the mixup loss's negligible influence on classifier learning by maintaining mixed-label samples but removing the mixup loss.

---

>W4. Scalability

The most ideal condition for our proposed method is when the dataset size is greater than the square of the number of classes $(N \geq K^2)$. Under this condition, the imbalance can be completely resolved by generating an equal number of mixed samples for all mixed labels, and its effectiveness has been demonstrated on CIFAR10/100-LT, which satisfies this condition. However, as the reviewer pointed out, when the number of classes increases $(N < K^2)$, the balance of mixed labels cannot be fully guaranteed by our proposed method. Nevertheless, even under such conditions, our method still demonstrates competitive performance compared to other class-balanced samplers, as demonstrated by the results in Table 6 (Table 7 in the revised paper) in Appendix A. In Places-LT, both BMLS and MS achieve higher performance than other class-balanced samplers, and when BMLS alone is employed, it attains the highest performance on ImageNet-LT and competitive performance on iNaturalist2018, the most challenging benchmark considered in this work. Furthermore, the scaling experiment (Table 7) (Table 8 in the revised paper) demonstrates that the proposed method can be effectively handled through straightforward engineering adjustments.

This study identifies a novel research problem (does mixed label imbalance matter to minority collapse in imbalanced learning?), theoretically proves it (mixed label imbalance causes minority collapse), and proposes BMLS to address it along with the MS classifier to further reinforce it (balancing mixed labels mitigates minority collapse). The primary contribution lies in demonstrating that resolving this problem is a critical factor in improving model performance in empirical long-tailed settings. While we acknowledge the scalability issue in large-class settings, Appendix A demonstrates that our works retains its effectiveness to a reasonable degree in large-class settings (Places-LT, ImageNet-LT), and the scaling experiment on ImageNet-LT highlights its potential for further development. We hope that this additional effort, which establishes a foundation for future investigation, will be taken into consideration.

---

>W5. Mismatch between the theoretical coverage $(K^2)$ and the practical method $(a<b)$

While we aim to generate an equal number of mixed samples for all $K^2$ mixed labels, in cases where this is not feasible $(N < K^2)$, mixed samples are generated only for randomly selected mixed labels and used for training. In practice, this is how the method operates on Places-LT, ImageNet-LT, and iNaturalist2018, where BMLS has been shown to be effective to a reasonable degree. Furthermore, the proposed $a < b$ method should be viewed not as a discrepancy with the theory, but as a special case contained within it. Specifically, $a < b$ method is derived from a special case of the Theorem 4.3, motivated by two questions: whether mixed labels $(a, b)$ and $(b, a)$ can be treated as equivalent, and whether training with only $(a, b)$ without singleton labels still forms a simplex ETF. Through our proof in Appendix D.6, we demonstrate that even when only $a < b$ exists, the existing assumptions for all singleton classes (i.e., satisfying NC and forming a simplex ETF) remain intact, and thus the proposed Theorem 4.3 is still satisfied. For this reason, it is difficult to regard the theoretical coverage (all $K^2$) and the practical method ($a < b$) as mismatched.

---

>W6. The empirical validation of MS

Table 3 was designed under such a setting to highlight that MS enhances the minority collapse mitigation of BMLS. However, as the reviewer pointed out, showcasing the independent contribution of MS in isolation is of significance, and this has already been addressed in our ablation study (Table 5). To better highlight this aspect, we added a diff row to the table to more clearly illustrate the performance differences.

---

>W7 & Q3. Indirect evidence for the claimed minority collapse mechanism

We appreciate the reviewer's insightful comment. To furnish more direct empirical evidence, we incorporate traditional NC metrics (NC1:Within-class Variability, NC2: Simplex ETF Error, NC3: Self-duality Error) [A2,A3] to quantitatively measure and compare explicit collapse behavior for Mixup vs. BMLS vs. BMLS+MS on CIFAR10/100-LT. The results are presented in the table below, and the corresponding line graphs are available at the provided URL. 

| CIFAR10-LT | NC1  | NC2  | NC3  |
| ---------- | ---- | ---- | ---- |
| Mixup      | 0.17 | 0.74 | 0.43 |
| BMLS       | 0.25 | 0.52 | 0.38 |
| BMLS+MS    | 0.31 | 0.42 | 0.35 |

As evidenced by the results, NC2 and NC3, excluding NC1, exhibit lower values under the proposed method, suggesting that the proposed method more effectively preserves the simplex ETF structure. The higher NC1 value suggests that our proposed method produces a stronger augmentation effect. This aligns with the observation that higher NC1 values are associated with greater representational diversity and improved expressiveness in OOD settings [A4].

[A1] Fisher et al., Pushing Boundaries: Mixup's Influence on Neural Collapse, ICLR'2024

[A2] Paypan et al., Prevalence of neural collapse during the terminal phase of deep learning training, PNAS'2020

[A3] https://github.com/rhubarbwu/neural-collapse

[A4] Harun et al., Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning, ICML'2025


### A. Finding BSCs satisfying \(F_{BSC(p)} \subseteq F_{WOM(\alpha, \epsilon)}\)

To determine when the condition \(F_{BSC(p)} \subseteq F_{WOM(\alpha, \epsilon)}\) holds, we conducted experiments with polar codes of length \(N = 8192\). The frozen sets were chosen based on the method in [10], with a rate loss \(\Delta R = 0.025\) for the WOM channel and a target block error rate of \(10^{-5}\) for BSC(p). The results, shown in Figure 7, indicate that the maximum value of \(p\) satisfying this condition increases with both \(\alpha\) and \(\epsilon\), suggesting practical applicability for typical memory error probabilities.

### B. Achievable Sum-Rates

We explored the achievable sum-rates for t-write error-correcting WOM codes by optimizing parameters \(\epsilon_1, \epsilon_2, \ldots, \epsilon_t\). For \(t = 2, 3, 4, 5\) writes, and with \(N = 8192\), \(\Delta R = 0.025\), and a target block error rate of \(10^{-5}\), the results in Figure 8 show that the achievable sum-rate increases with \(t\). Notably, the rates for the general code are close to those of the nested code, indicating that the nested structure performs well within this parameter range. The lower bound on sum-rate from Figure 6 is higher due to the infinite length assumption and rate loss considerations.

### C. Code Analysis for BSC

We proved the correctness of our code construction by demonstrating that the encoder rewrites data effectively while the decoder recovers the message despite noise. Key insights include the equivalence of noisy channels for cell levels \(s\) and values \(v\). Lemma 1 provides bounds on the intersection of frozen sets, showing how \(\alpha\) and \(\epsilon\) affect the relationship between BSC(p) and WOM(α, ε) channels. These findings are crucial for understanding the code's performance and simplifying its design.

### D. Experimental Setup

Our experiments were conducted with polar codes of length \(N = 8192\), using a rate loss \(\Delta R = 0.025\) and a target block error rate of \(10^{-5}\). The frozen sets for the WOM channel were chosen to match the theoretical capacity, while those for BSC(p) were selected to achieve the desired error performance. By varying parameters such as \(\alpha\) and \(\epsilon\), we explored different scenarios to find optimal configurations that maximize sum-rate. The results provide valuable insights into the practicality and efficiency of our error-correcting WOM codes.
